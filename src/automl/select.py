"""AutoML: автоматический выбор оптимальной DL архитектуры.

Программа максимум: по характеристикам датасета выбрать лучшую архитектуру.

Пайплайн:
  1. Загрузить подвыборку датасета (например, 1000 молекул).
  2. Извлечь геометрические priors через TDA (src.tda.priors).
  3. Рекомендовать архитектуру (src.tda.priors.recommend_architecture).
  4. Опционально: quick-train candidate-моделей на подвыборке и сравнить val_loss.
  5. Выбрать лучшую и сохранить отчёт.

Использование:
  python src/automl/select.py \\
      --data_dir data/alchemy \\
      --n_molecules 100 \\
      --output_json results/automl/recommendation.json

  # С quick-train сравнением:
  python src/automl/select.py \\
      --data_dir data/alchemy \\
      --n_molecules 500 \\
      --epochs 3 \\
      --quick_train \\
      --output_json results/automl/recommendation.json
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="AutoML: choose best architecture via TDA priors")
    p.add_argument("--data_dir", type=str, default="data/alchemy")
    p.add_argument("--n_molecules", type=int, default=100,
                   help="Сколько молекул взять для extraction priors")
    p.add_argument("--n_bins", type=int, default=16)
    p.add_argument("--max_radius", type=float, default=5.0)
    p.add_argument("--n_trials", type=int, default=3,
                   help="Число случайных преобразований на молекулу")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.9,
                   help="Score >= threshold считается сильным prior")
    p.add_argument("--quick_train", action="store_true",
                   help="Дополнительно quick-train несколько моделей и сравнить")
    p.add_argument("--epochs", type=int, default=3,
                   help="Эпох для quick_train")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output_json", type=str, default=None,
                   help="Куда сохранить отчёт. По умолчанию results/automl/recommendation.json")
    p.add_argument("--candidates", type=str, default="fcnn,schnet,egnn,egnn_tda",
                   help="Список моделей для quick_train через запятую")
    return p.parse_args()


def load_molecules_for_priors(data_dir: str, n_molecules: int, seed: int = 42):
    """Загрузить n_molecules молекул (только координаты) для extraction priors."""
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.data import find_sdf_files, load_properties_csv, mol_to_arrays, parse_sdf

    data_root = Path(data_dir) / "Alchemy-v20191129"
    csv_path = data_root / "final_version.csv"

    print(f"Loading properties from {csv_path} ...")
    props = load_properties_csv(csv_path)
    print(f"  Properties: {len(props)} molecules")

    print(f"Finding SDF files in {data_root} ...")
    sdf_files = find_sdf_files(data_root)
    print(f"  Found: {len(sdf_files)} SDF files")

    valid_gdb = sorted(set(sdf_files.keys()) & set(props["gdb_idx"].tolist()))
    print(f"  Valid: {len(valid_gdb)}")

    # Берём подвыборку
    rng = np.random.default_rng(seed)
    if n_molecules < len(valid_gdb):
        indices = rng.choice(valid_gdb, size=n_molecules, replace=False)
    else:
        indices = valid_gdb

    coords_list = []
    for i, gdb_idx in enumerate(indices):
        if i % max(1, len(indices) // 10) == 0:
            print(f"  Parsing SDF: {i}/{len(indices)}")
        mol = parse_sdf(sdf_files[gdb_idx])
        if mol is None:
            continue
        arr = mol_to_arrays(mol)
        coords_list.append(arr["pos"])

    print(f"  Loaded: {len(coords_list)} molecules")
    return coords_list


def quick_train_candidates(args, candidates: list[str]) -> dict:
    """Quick-train несколько моделей и вернуть их val_loss."""
    import subprocess
    results = {}

    output_dir = "results/automl/quick_train"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for model in candidates:
        print(f"\n{'='*60}")
        print(f"  Quick-training: {model}")
        print(f"{'='*60}")
        cmd = [
            sys.executable, str(Path(__file__).parent.parent / "train.py"),
            "--model", model, "--target", "all",
            "--epochs", str(args.epochs),
            "--max_train", str(args.n_molecules),
            "--max_val", str(max(50, args.n_molecules // 5)),
            "--max_test", str(max(50, args.n_molecules // 5)),
            "--batch_size", str(args.batch_size),
            "--device", args.device,
            "--num_workers", "0",  # CPU без workers для стабильности
            "--output_dir", output_dir,
            "--es_mode", "or",  # быстро стопаем
            "--patience", str(args.epochs),  # по сути без ES
        ]
        if model in ("egnn_tda", "egnn_vector_tda"):
            cmd.extend(["--n_bins", str(args.n_bins)])

        print(f"  Command: {' '.join(cmd[:6])} ...")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if proc.returncode == 0:
                # Найти последний history CSV для этой модели
                import glob
                import os
                csvs = sorted(glob.glob(f"{output_dir}/history_{model}_all_*.csv"),
                              key=os.path.getmtime)
                if csvs:
                    import pandas as pd
                    df = pd.read_csv(csvs[-1])
                    best_val = float(df["val_loss"].min())
                    last_test = df.iloc[-1].get("test_loss", float("nan"))
                    results[model] = {
                        "status": "OK",
                        "best_val_loss": round(best_val, 4),
                        "test_loss": round(float(last_test), 4) if not pd.isna(last_test) else None,
                        "epochs": len(df),
                        "csv": os.path.basename(csvs[-1]),
                    }
                    print(f"  [OK] {model}: best_val_loss={best_val:.4f}")
                else:
                    results[model] = {"status": "OK but no CSV", "best_val_loss": None}
            else:
                results[model] = {"status": f"FAIL exit={proc.returncode}",
                                  "stderr": proc.stderr[-500:]}
                print(f"  [FAIL] {model}: exit={proc.returncode}")
        except subprocess.TimeoutExpired:
            results[model] = {"status": "TIMEOUT"}
            print(f"  [TIMEOUT] {model}")
        except Exception as e:
            results[model] = {"status": f"ERROR: {e}"}
            print(f"  [ERROR] {model}: {e}")

    return results


def main():
    args = parse_args()

    print("=" * 60)
    print("  AutoML: архитектурный выбор через TDA priors")
    print("=" * 60)

    # === Шаг 1: загрузка молекул ===
    print("\n[1/4] Загрузка молекул для extraction priors ...")
    coords_list = load_molecules_for_priors(args.data_dir, args.n_molecules, args.seed)

    # === Шаг 2: extraction priors ===
    print("\n[2/4] Извлечение геометрических priors ...")
    from src.tda.priors import extract_priors, recommend_architecture
    priors = extract_priors(
        coords_list,
        n_bins=args.n_bins,
        max_radius=args.max_radius,
        n_trials=args.n_trials,
        seed=args.seed,
        verbose=True,
    )

    # === Шаг 3: рекомендация архитектуры ===
    print("\n[3/4] Рекомендация архитектуры ...")
    recommendation = recommend_architecture(priors, threshold=args.threshold)
    print(f"\nRequired invariances: {recommendation['required_invariances']}")
    print(f"Recommended model: {recommendation['recommended_model']}")
    print(f"Needs TDA: {recommendation['needs_tda']}")
    print(f"Rationale: {recommendation['rationale']}")

    # === Шаг 4: опционально quick-train ===
    quick_train_results = None
    if args.quick_train:
        print("\n[4/4] Quick-train сравнение candidate-моделей ...")
        candidates = args.candidates.split(",")
        quick_train_results = quick_train_candidates(args, candidates)

        # Найти лучшую по best_val_loss
        valid = {k: v for k, v in quick_train_results.items()
                 if v.get("best_val_loss") is not None}
        if valid:
            best_model = min(valid, key=lambda k: valid[k]["best_val_loss"])
            print(f"\nBest model by quick_train: {best_model} "
                  f"(val_loss={valid[best_model]['best_val_loss']})")
            recommendation["quick_train_best"] = best_model
            recommendation["quick_train_best_val_loss"] = valid[best_model]["best_val_loss"]

    # === Сохранение отчёта ===
    output_json = args.output_json or "results/automl/recommendation.json"
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "priors": priors,
        "recommendation": recommendation,
        "quick_train_results": quick_train_results,
    }
    # Убираем per_molecule из priors для краткости отчёта
    report["priors"].pop("per_molecule", None)

    with open(output_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n=== Отчёт сохранён: {output_json} ===")

    # Печать финальной рекомендации
    print("\n" + "=" * 60)
    print("  ФИНАЛЬНАЯ РЕКОМЕНДАЦИЯ")
    print("=" * 60)
    print(f"  Recommended model: {recommendation['recommended_model']}")
    print(f"  Required invariances: {recommendation['required_invariances']}")
    print(f"  Needs TDA: {recommendation['needs_tda']}")
    if "quick_train_best" in recommendation:
        print(f"  Quick-train best: {recommendation['quick_train_best']} "
              f"(val_loss={recommendation['quick_train_best_val_loss']})")
    print()
    print(f"  Rationale: {recommendation['rationale']}")


if __name__ == "__main__":
    main()
