"""Robustness evaluation: прогон обученной модели при разном уровне шума в координатах.

Загружает чекпойнт модели, прогоняет evaluate() при sigma ∈ {0.0, 0.05, 0.10, 0.15}
(настраивается через --noise_sigma), записывает таблицу в CSV.

Использование:
  # Запуск для одной модели
  python src/eval_robustness.py \\
      --model egnn \\
      --target all \\
      --checkpoint checkpoints/egnn_all_best.pt \\
      --target_stats results/experiments/batch_size_512/target_stats_egnn_all.json \\
      --output_csv results/robustness/egnn_robustness.csv

  # Несколько значений sigma
  python src/eval_robustness.py --model egnn --target all \\
      --checkpoint ckpt.pt --noise_sigma 0.0,0.05,0.10,0.15,0.20
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from tda.features import tda_feature_dim
from train import build_model, evaluate
from utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Robustness evaluation")
    p.add_argument("--model", type=str, required=True,
                   choices=["fcnn", "schnet", "egnn", "egnn_tda", "egnn_vector", "egnn_vector_tda"])
    p.add_argument("--target", type=str, default="all",
                   choices=["mu", "alpha", "gap", "all"])
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Путь к .pt чекпойнту")
    p.add_argument("--target_stats", type=str, required=True,
                   help="Путь к target_stats_<model>_<target>.json (сохраняется train.py)")
    p.add_argument("--data_dir", type=str, default="data/alchemy")
    p.add_argument("--output_csv", type=str, default=None,
                   help="Куда сохранить CSV с результатами. "
                        "По умолчанию: results/robustness/<model>_robustness.csv")
    p.add_argument("--noise_sigma", type=str, default="0.0,0.05,0.10,0.15",
                   help="Список sigma через запятую")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--hidden_channels", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--k_neighbors", type=int, default=16)
    p.add_argument("--m_dim", type=int, default=32)
    p.add_argument("--n_bins", type=int, default=16)
    p.add_argument("--tda_mode", type=str, default="concat",
                   choices=["concat", "film"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Загрузка target_stats
    with open(args.target_stats) as f:
        target_stats_raw = json.load(f)
    target_stats = {k: tuple(v) for k, v in target_stats_raw.items()}
    print(f"Loaded target_stats: {target_stats}")

    # Загрузка датасета (только test split)
    use_tda = args.model in ("egnn_tda", "egnn_vector_tda")
    print(f"Loading test dataset (use_tda={use_tda}) ...")
    from dataset import AlchemyDataset
    test_ds = AlchemyDataset(
        root=args.data_dir, split="test",
        tda_features=use_tda, n_bins=args.n_bins,
        max_radius=args.cutoff, seed=args.seed,
    )
    print(f"Test molecules: {len(test_ds)}")

    from torch_geometric.loader import DataLoader as PyGDataLoader
    loader_kwargs = {"batch_size": args.batch_size}
    if device.type == "cuda":
        loader_kwargs["num_workers"] = args.num_workers
        loader_kwargs["pin_memory"] = True
        loader_kwargs["persistent_workers"] = args.num_workers > 0
    test_loader = PyGDataLoader(test_ds, shuffle=False, **loader_kwargs)

    # Построение и загрузка модели
    tda_dim = tda_feature_dim(args.n_bins) if use_tda else 0
    model = build_model(args, tda_dim=tda_dim).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Прогон при разных sigma
    sigmas = [float(s) for s in args.noise_sigma.split(",")]
    print(f"\nWill evaluate at sigma = {sigmas}")

    rows = []
    for sigma in sigmas:
        print(f"\n{'='*60}")
        print(f"  sigma = {sigma}")
        print(f"{'='*60}")
        # Устанавливаем noise на test через args
        args.noise = sigma
        args.noise_mode = "test_only"  # только test получает шум

        # Имитируем args-объект для evaluate
        class _ArgsNS:
            pass
        eval_args = _ArgsNS()
        for attr in ["target", "noise", "noise_mode"]:
            setattr(eval_args, attr, getattr(args, attr))

        test_metrics = evaluate(
            model, test_loader, device, eval_args,
            logger=None, prefix="test",
            target_stats=target_stats, use_multi_gpu=False,
        )
        print(f"  Test metrics: {test_metrics}")

        row = {"sigma": sigma}
        for k, v in test_metrics.items():
            row[f"test_{k}"] = v
        rows.append(row)

    # Сохранение CSV
    import csv
    output_csv = args.output_csv or f"results/robustness/{args.model}_robustness.csv"
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n=== Robustness CSV сохранён: {output_csv} ===")

    # Печать таблицы
    print(f"\n{'sigma':>8s} | {'mu_mae':>10s} | {'alpha_mae':>10s} | {'gap_mae':>10s} | {'loss':>10s}")
    print("-" * 60)
    for row in rows:
        print(f"{row['sigma']:>8.3f} | {row.get('test_mu_mae', 0):>10.4f} | "
              f"{row.get('test_alpha_mae', 0):>10.4f} | "
              f"{row.get('test_gap_mae', 0):>10.4f} | {row.get('test_loss', 0):>10.4f}")


if __name__ == "__main__":
    main()
