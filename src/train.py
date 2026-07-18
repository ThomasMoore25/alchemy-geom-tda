"""Основной скрипт обучения.

Поддерживаемые модели: fcnn, schnet,
                      egnn, egnn_tda, egnn_vector, egnn_vector_tda

Примеры запуска:
  # EGNN (основная модель)
  python src/train.py --model egnn --target all --epochs 100

  # EGNN + TDA
  python src/train.py --model egnn_tda --target all --epochs 100

  # EGNN с векторным выходом mu
  python src/train.py --model egnn_vector --target all --epochs 100

  # Multi-GPU (PyG DataParallel)
  python src/train.py --model egnn --target all --multi_gpu --num_workers 4

  # Для отладки (на 15 молекулах)
  python src/train.py --model egnn --target all --epochs 5 --max_train 15

  # Оценка на зашумлённых координатах
  python src/train.py --model egnn_tda --target all --eval_only \\
      --checkpoint checkpoints/egnn_tda_all_best.pt --noise 0.10
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))

from tda.features import tda_feature_dim
from utils import seed_everything, setup_logger


def _load_yaml_config(path: str) -> dict:
    """Загрузить YAML-конфиг. Возвращает плоский dict аргументов для argparse."""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    flat = {}
    # experiment
    if "experiment" in cfg:
        if "seed" in cfg["experiment"]:
            flat["seed"] = cfg["experiment"]["seed"]
        if "device" in cfg["experiment"]:
            flat["device"] = cfg["experiment"]["device"]
    # data
    if "data" in cfg:
        d = cfg["data"]
        if "data_dir" in d:
            flat["data_dir"] = d["data_dir"]
        if "batch_size" in d:
            flat["batch_size"] = d["batch_size"]
        if "num_workers" in d:
            flat["num_workers"] = d["num_workers"]
        if d.get("max_train") is not None:
            flat["max_train"] = d["max_train"]
        if d.get("max_val") is not None:
            flat["max_val"] = d["max_val"]
        if d.get("max_test") is not None:
            flat["max_test"] = d["max_test"]
    # tda
    if "tda" in cfg:
        if "n_bins" in cfg["tda"]:
            flat["n_bins"] = cfg["tda"]["n_bins"]
        if "max_radius" in cfg["tda"]:
            flat["max_radius"] = cfg["tda"]["max_radius"]
    # model
    if "model" in cfg:
        m = cfg["model"]
        if "name" in m:
            flat["model"] = m["name"]
        if "hidden_channels" in m:
            flat["hidden_channels"] = m["hidden_channels"]
        if "num_layers" in m:
            flat["num_layers"] = m["num_layers"]
        if "num_rbf" in m:
            flat["num_rbf"] = m["num_rbf"]
        if "cutoff" in m:
            flat["cutoff"] = m["cutoff"]
    # training
    if "training" in cfg:
        t = cfg["training"]
        for k_cli, k_yaml in [
            ("epochs", "epochs"), ("lr", "lr"), ("weight_decay", "weight_decay"),
            ("grad_clip", "grad_clip"), ("lr_patience", "lr_patience"),
            ("patience", "patience"),
        ]:
            if k_yaml in t:
                flat[k_cli] = t[k_yaml]
    # targets
    if "targets" in cfg and "target" in cfg["targets"]:
        flat["target"] = cfg["targets"]["target"]
    return flat


def parse_args():
    # Сначала парсим только --config, чтобы подгрузить YAML-дефолты.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_known, remaining = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Alchemy GeomML + TDA training")
    p.add_argument("--config", type=str, default=None,
                   help="Путь к YAML-конфигу (например, configs/default.yaml). "
                        "Значения из CLI имеют приоритет над значениями из YAML.")
    p.add_argument("--model", type=str, default=None,
                   choices=["fcnn", "schnet", "egnn", "egnn_tda", "egnn_vector", "egnn_vector_tda",
                            "egnn_tensor"],
                   help="Тип модели")
    p.add_argument("--target", type=str, default="all",
                   choices=["mu", "alpha", "gap", "all"],
                   help="Целевое свойство")
    p.add_argument("--data_dir", type=str, default="data/alchemy")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--output_dir", type=str, default="results",
                   help="Куда складывать CSV-истории (по умолчанию results/)")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=9999,
                   help="Максимум эпох (EarlyStopping остановит раньше)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate. 1e-3 — canonical default для EGNN.")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden_channels", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4,
                   help="Количество EGNN-слоёв")
    p.add_argument("--cutoff", type=float, default=5.0,
                   help="Радиус отсечения и нормализация координат (pos / cutoff)")
    p.add_argument("--k_neighbors", type=int, default=16,
                   help="Число соседей в kNN-графе для EGNN-моделей")
    p.add_argument("--m_dim", type=int, default=32,
                   help="Размерность m в EGNN_Sparse")
    p.add_argument("--noise", type=float, default=0.0,
                   help="Шум в координатах (для robustness test)")
    p.add_argument("--noise_mode", type=str, default="test_only",
                   choices=["test_only", "train_val_test", "train_only", "eval_only"],
                   help="Куда добавлять шум: "
                        "test_only (default, v31 behavior) — только test, "
                        "train_val_test — train+val+test (consistent eval), "
                        "train_only — только train (data augmentation), "
                        "eval_only — val+test (no train augmentation)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_only", action="store_true",
                   help="Только оценка (нужен --checkpoint)")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="auto",
                   help="Device: 'auto' (default, использует cuda если доступна), "
                        "'cpu', 'cuda', или 'cuda:N' для конкретной GPU (например, 'cuda:0')")
    p.add_argument("--max_train", type=int, default=None,
                   help="Лимит числа обучающих молекул (для отладки)")
    p.add_argument("--max_val", type=int, default=None,
                   help="Лимит валидационных молекул (для отладки)")
    p.add_argument("--max_test", type=int, default=None,
                   help="Лимит тестовых молекул (для отладки)")
    p.add_argument("--n_bins", type=int, default=16, help="TDA Betti bins")
    p.add_argument("--max_radius", type=float, default=5.0, help="TDA радиус")
    p.add_argument("--tda_mode", type=str, default="concat",
                   choices=["concat", "film"],
                   help="Способ интеграции TDA: concat (по умолчанию) или film "
                        "(FiLM-модуляция mol_emb через TDA)")
    p.add_argument("--predict_tensor_alpha", action="store_true",
                   help="Часть B: предсказывать полный тензор поляризуемости α ∈ R^(3×3) "
                        "вместо скалярной изотропной α. Только для egnn_tensor модели.")
    p.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    p.add_argument("--es_mode", type=str, default="and",
                   choices=["and", "or", "loss_only"],
                   help="Early stopping режим: "
                        "and (default) — все метрики должны перестать улучшаться, "
                        "or — хотя бы одна метрика перестала улучшаться, "
                        "loss_only — только val_loss отслеживается")
    p.add_argument("--min_delta", type=float, default=0.0, help="Early stopping min delta")
    p.add_argument("--lr_patience", type=int, default=5, help="ReduceLROnPlateau patience")
    p.add_argument("--multi_gpu", action="store_true",
                   help="Включить PyG DataParallel для multi-GPU (v29). "
                        "По умолчанию off — обучение на 1 GPU.")
    p.add_argument("--num_workers", type=int, default=4,
                   help="Кол-во worker процессов для DataLoader (v30). "
                        "0 = без workers (медленно). По умолчанию 4.")

    # YAML как default: CLI > YAML > argparse default.
    if pre_known.config is not None:
        yaml_cfg = _load_yaml_config(pre_known.config)
        # set_defaults переопределяет argparse-дефолт, но CLI переопределит yaml
        p.set_defaults(**yaml_cfg)

    args = p.parse_args()
    if args.model is None:
        p.error("--model обязателен (через CLI или через --config с model.name)")
    return args


def build_model(args, tda_dim: int = 0):
    """Создать модель по аргументам."""
    pred_mu = args.target in ("mu", "all")
    pred_alpha = args.target in ("alpha", "all")
    pred_gap = args.target in ("gap", "all")

    if args.model == "fcnn":
        from models.fcnn import build_fcnn
        out_dim = 3 if args.target == "all" else 1
        return build_fcnn(in_dim=8 * 3, out_dim=out_dim,
                          hidden_dim=args.hidden_channels, n_layers=args.num_layers)

    elif args.model == "schnet":
        from models.schnet import build_schnet
        out_dim = 3 if args.target == "all" else 1
        return build_schnet(out_dim=out_dim, hidden_channels=args.hidden_channels,
                            num_interactions=args.num_layers, cutoff=args.cutoff)

    elif args.model == "egnn":
        from models.egnn import build_egnn
        return build_egnn(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            cutoff=args.cutoff,
            k_neighbors=args.k_neighbors,
            m_dim=args.m_dim,
            predict_mu=pred_mu,
            predict_alpha=pred_alpha,
            predict_gap=pred_gap,
        )

    elif args.model == "egnn_tda":
        from models.egnn_tda import build_egnn_tda
        return build_egnn_tda(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            cutoff=args.cutoff,
            k_neighbors=args.k_neighbors,
            m_dim=args.m_dim,
            tda_dim=tda_dim or tda_feature_dim(args.n_bins),
            tda_mode=args.tda_mode,
            predict_mu=pred_mu,
            predict_alpha=pred_alpha,
            predict_gap=pred_gap,
        )

    elif args.model == "egnn_vector":
        from models.egnn_vector import build_egnn_vector
        return build_egnn_vector(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            cutoff=args.cutoff,
            k_neighbors=args.k_neighbors,
            m_dim=args.m_dim,
            predict_alpha=pred_alpha,
            predict_gap=pred_gap,
        )

    elif args.model == "egnn_vector_tda":
        from models.egnn_vector_tda import build_egnn_vector_tda
        return build_egnn_vector_tda(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            cutoff=args.cutoff,
            k_neighbors=args.k_neighbors,
            m_dim=args.m_dim,
            tda_dim=tda_dim or tda_feature_dim(args.n_bins),
            tda_mode=args.tda_mode,
            predict_alpha=pred_alpha,
            predict_gap=pred_gap,
        )

    elif args.model == "egnn_tensor":
        # Часть B: вектор μ + тензор α (программа максимума)
        from models.egnn_tensor import build_egnn_tensor
        return build_egnn_tensor(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            cutoff=args.cutoff,
            k_neighbors=args.k_neighbors,
            m_dim=args.m_dim,
            predict_alpha_tensor=args.predict_tensor_alpha,
            predict_gap=pred_gap,
        )

    raise ValueError(f"Unknown model: {args.model}")


def _unpack_preds(preds, target: str) -> dict:
    """Унификация: FCNN/SchNet возвращают тензор, EGNN — словарь."""
    if isinstance(preds, dict):
        return preds
    if target == "all":
        return {"mu": preds[:, 0:1], "alpha": preds[:, 1:2], "gap": preds[:, 2:3]}
    return {target: preds}


def _get_target(key: str, batch, target_stats: dict | None = None):
    """Получить таргет по ключу. Для векторного mu — возвращаем как есть (B, 3).
    Для скаляров — нормализуем если есть target_stats."""
    val = getattr(batch, key)
    if target_stats is not None and key in target_stats:
        m, s = target_stats[key]
        return (val - m) / s
    return val


def compute_loss(preds, batch, target: str, target_stats: dict | None = None) -> torch.Tensor:
    """Вычислить loss. Для векторного mu (B,3) — сравниваем норму со скалярным таргетом.

    v32: добавлен clamp(min=eps) при взятии нормы вектора, чтобы избежать
    сингулярности градиента d|x|/dx = x/|x| при |x| -> 0.

    v32: если pred векторный (B,3) и есть target_stats, нормализуем pred.norm()
    теми же mean/std, что и target.

    v33 (часть B): если в preds есть 'alpha_tensor' (B, 3, 3), добавляем
    soft regularization на симметрию (хотя по построению уже симметричный,
    на всякий случай штрафуем ||α − αᵀ||₂).
    """
    preds = _unpack_preds(preds, target)

    # v33.8: egnn_tensor возвращает физические μ и α (не нормализованные).
    # Признак: наличие 'alpha_tensor' в preds.
    is_physics_model = "alpha_tensor" in preds

    loss = 0.0
    for key in ["mu", "alpha", "gap"]:
        if target not in (key, "all"):
            continue
        if key not in preds:
            continue
        pred_val = preds[key]
        target_val = _get_target(key, batch, target_stats)

        # Если pred векторный (B,3) — берём норму с clamp для стабильности градиента
        if pred_val.dim() == 2 and pred_val.shape[1] == 3:
            pred_val = pred_val.norm(dim=-1, keepdim=True).clamp(min=1e-4)
            # v33.8: нормализуем pred.norm() через target_stats
            if target_stats is not None and key in target_stats:
                m, s = target_stats[key]
                pred_val = (pred_val - m) / s
        # v33.8: для физической модели (egnn_tensor) скалярная alpha тоже
        # в физических единицах — нормализуем через target_stats
        elif is_physics_model and key == "alpha" and target_stats is not None and key in target_stats:
            m, s = target_stats[key]
            pred_val = (pred_val - m) / s

        # Если target одномерный (B,) — добавляем размерность
        if target_val.dim() == 1:
            target_val = target_val.unsqueeze(-1)

        loss = loss + (pred_val - target_val).abs().mean()

    # v33 (часть B): regularization на симметрию тензора поляризуемости
    if "alpha_tensor" in preds:
        alpha_t = preds["alpha_tensor"]  # (B, 3, 3)
        sym_reg = (alpha_t - alpha_t.transpose(-1, -2)).pow(2).mean()
        loss = loss + 0.01 * sym_reg  # малый вес — regularization

    return loss


def compute_metrics(preds, batch, target: str, target_stats: dict | None = None,
                     as_item: bool = True) -> dict:
    """Вычислить метрики в исходных единицах.

    v30: as_item=False возвращает тензоры на GPU (без .item() синхронизации).
    Используется в train loop для отложенной синхронизации.
    """
    preds = _unpack_preds(preds, target)
    metrics = {}

    # v33.8: egnn_tensor возвращает физические μ и α (не нормализованные).
    is_physics_model = "alpha_tensor" in preds

    for key in ["mu", "alpha", "gap"]:
        if target not in (key, "all"):
            continue
        if key not in preds:
            continue
        pred_val = preds[key]
        target_val = getattr(batch, key)

        # Если target одномерный — добавляем размерность
        if target_val.dim() == 1:
            target_val = target_val.unsqueeze(-1)

        # Денормализуем предсказание
        if target_stats is not None and key in target_stats:
            mean, std = target_stats[key]
            if pred_val.dim() == 2 and pred_val.shape[1] == 3:
                # Векторный mu (egnn_vector, egnn_tensor):
                # pred = |μ_pred| в физических единицах (Дебай)
                # Денормализация: pred * std + mean
                pred_val = pred_val.norm(dim=-1, keepdim=True)
                pred_val = pred_val * std + mean
            elif is_physics_model and key == "alpha":
                # v33.8: для egnn_tensor, alpha = polarizability_iso — уже
                # в физических единицах (Боровские кубы). Денормализация НЕ нужна.
                pass
            else:
                # Обычная модель (EGNN, SchNet, FCNN): pred — нормализованное
                # значение из MLP head. Денормализация: pred * std + mean
                pred_val = pred_val * std + mean

        # Если pred всё ещё векторный (после денормализации не должно быть)
        if pred_val.dim() == 2 and pred_val.shape[1] == 3:
            pred_val = pred_val.norm(dim=-1, keepdim=True)

        mae_tensor = (pred_val - target_val).abs().mean()
        metrics[f"{key}_mae"] = mae_tensor.item() if as_item else mae_tensor
    return metrics


def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    # v32: валидация cuda:N — если указан конкретный индекс, проверить что он существует
    if device.type == "cuda" and device.index is not None:
        if device.index >= n_gpus:
            raise ValueError(
                f"Запрошена cuda:{device.index}, но доступно только {n_gpus} GPU "
                f"(индексы 0..{n_gpus-1}). Доступные устройства: "
                f"{[torch.cuda.get_device_name(i) for i in range(n_gpus)]}"
            )
        print(f"Selected GPU {device.index}: {torch.cuda.get_device_name(device.index)}")
    print(f"Device: {device}" + (f"  (GPUs: {n_gpus})" if n_gpus > 0 else ""))

    logger = setup_logger("train", log_file=f"logs/{args.model}_{args.target}.log")
    # Загрузка датасета
    logger.info("Загрузка датасета Alchemy...")
    from dataset import AlchemyDataset

    use_tda = args.model in ("egnn_tda", "egnn_vector_tda")
    train_ds = AlchemyDataset(root=args.data_dir, split="train",
                              max_samples=args.max_train,
                              tda_features=use_tda, n_bins=args.n_bins,
                              max_radius=args.max_radius, seed=args.seed)
    val_ds = AlchemyDataset(root=args.data_dir, split="val",
                            max_samples=args.max_val,
                            tda_features=use_tda, n_bins=args.n_bins,
                            max_radius=args.max_radius, seed=args.seed)
    test_ds = AlchemyDataset(root=args.data_dir, split="test",
                             max_samples=args.max_test,
                             tda_features=use_tda, n_bins=args.n_bins,
                             max_radius=args.max_radius, seed=args.seed)
    logger.info(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    # === Нормализация таргетов (по train выборке) ===
    # Считаем mean/std для mu, alpha, gap и храним в словаре
    target_stats = {}
    for key in ["mu", "alpha", "gap"]:
        vals = torch.cat([getattr(d, key) for d in train_ds])
        target_stats[key] = (float(vals.mean()), float(vals.std() + 1e-8))
    logger.info(f"Target stats (mean, std): {target_stats}")

    # v32: сохраняем args и target_stats для воспроизводимости
    # Без них чекпойнт бесполезен: неизвестно, как денормализовать pred,
    # неизвестны гиперпараметры обучения.
    import json
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args_path = Path(args.output_dir) / f"args_{args.model}_{args.target}.json"
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    logger.info(f"→ args сохранены в {args_path}")

    stats_path = Path(args.output_dir) / f"target_stats_{args.model}_{args.target}.json"
    with open(stats_path, "w") as f:
        json.dump({k: list(v) for k, v in target_stats.items()}, f, indent=2)
    logger.info(f"→ target_stats сохранены в {stats_path}")

    from torch_geometric.data import Batch
    from torch_geometric.loader import DataListLoader
    from torch_geometric.loader import DataLoader as PyGDataLoader

    # v29: PyG DataParallel требует DataListLoader (возвращает list[Data] вместо Batch)
    use_multi_gpu = (args.multi_gpu and torch.cuda.device_count() > 1
                     and device.type == "cuda")
    LoaderCls = DataListLoader if use_multi_gpu else PyGDataLoader
    logger.info(f"Loader: {LoaderCls.__name__}  (multi_gpu={use_multi_gpu})")

    # v30: num_workers + pin_memory для ускорения загрузки батчей
    # DataListLoader (multi-GPU) не поддерживает num_workers корректно с PyG,
    # поэтому workers только для 1-GPU режима
    loader_kwargs = {"batch_size": args.batch_size}
    if not use_multi_gpu and device.type == "cuda":
        loader_kwargs["num_workers"] = args.num_workers
        loader_kwargs["pin_memory"] = True
        loader_kwargs["persistent_workers"] = args.num_workers > 0
    logger.info(f"Loader kwargs: {loader_kwargs}")

    train_loader = LoaderCls(train_ds, shuffle=True, **loader_kwargs)
    val_loader = LoaderCls(val_ds, shuffle=False, **loader_kwargs)
    test_loader = LoaderCls(test_ds, shuffle=False, **loader_kwargs)

    tda_dim = tda_feature_dim(args.n_bins) if args.model in ("egnn_tda", "egnn_vector_tda") else 0

    model = build_model(args, tda_dim=tda_dim).to(device)
    # v29: PyG DataParallel (заменяет неработающий nn.DataParallel из v27/v28)
    if use_multi_gpu:
        from torch_geometric.nn import DataParallel as PyGDataParallel
        n_gpu = torch.cuda.device_count()
        logger.info(f"Использую PyG DataParallel на {n_gpu} GPU")
        model = PyGDataParallel(model)
        _underlying = model.module
    else:
        _underlying = model
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Модель: {args.model}, параметров: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_patience)

    if args.eval_only:
        if args.checkpoint is None:
            logger.error("--eval_only требует --checkpoint")
            return
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        test_metrics = evaluate(model, test_loader, device, args, logger, prefix="test")
        for k, v in test_metrics.items():
            logger.info(f"  test_{k}: {v:.4f}")
        return

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)  # v27: output_dir вместо жёсткого results
    ckpt_path = Path(args.checkpoint_dir) / f"{args.model}_{args.target}_best.pt"

    # === История обучения для графиков ===
    history = []

    # === Early Stopping (многопараметрическая) ===
    from early_stopping import EarlyStopping
    if args.es_mode == "loss_only":
        # Только val_loss — простой режим, как классический ES
        es_config = {'val_loss': 'min'}
        stop_mode = 'or'  # не имеет значения при одной метрике
    else:
        es_config = {'val_loss': 'min'}
        if args.target in ('mu', 'all'):
            es_config['val_mu_mae'] = 'min'
        if args.target in ('alpha', 'all'):
            es_config['val_alpha_mae'] = 'min'
        if args.target in ('gap', 'all'):
            es_config['val_gap_mae'] = 'min'
        stop_mode = args.es_mode  # 'and' or 'or'

    early_stopping = EarlyStopping(
        metrics_config=es_config,
        stop_mode=stop_mode,
        save_metric='val_loss',
        patience=args.patience,
        min_delta=args.min_delta,
    )
    logger.info(f"EarlyStopping: mode={args.es_mode}, "
                f"metrics={list(es_config.keys())}, patience={args.patience}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # === Train ===
        model.train()
        # v32: явная инициализация tensor на device для безопасности multi-GPU
        train_loss_sum = torch.zeros((), device=device)  # GPU тензор
        train_loss_count = 0
        train_metric_sums = {}  # v30: накапливаем GPU-тензоры
        train_counts = 0
        for batch_or_list in train_loader:
            # v29: унифицированный интерфейс — получаем Batch на device
            if use_multi_gpu:
                # DataListLoader возвращает list[Data]
                data_list = batch_or_list
                batch = Batch.from_data_list(data_list).to(device)
                num_graphs = len(data_list)
            else:
                batch = batch_or_list.to(device)
                data_list = None
                num_graphs = batch.num_graphs
            # v32: noise_mode управляет, куда добавлять шум
            if args.noise > 0 and args.noise_mode in ("train_val_test", "train_only"):
                batch.pos = batch.pos + torch.randn_like(batch.pos) * args.noise
            optimizer.zero_grad()
            preds = model(data_list) if use_multi_gpu else model(batch)
            loss = compute_loss(preds, batch, args.target, target_stats)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # v30 + v32: detach + явный .to(device) для multi-GPU безопасности
            train_loss_sum = train_loss_sum + loss.detach().to(device) * num_graphs
            train_loss_count += num_graphs

            with torch.no_grad():
                # v30: as_item=False — метрики как GPU-тензоры
                tr_metrics = compute_metrics(preds, batch, args.target, target_stats, as_item=False)
                for k, v in tr_metrics.items():
                    # v32: явно на device для multi-GPU
                    v_dev = v.detach().to(device) * num_graphs
                    if k not in train_metric_sums:
                        train_metric_sums[k] = v_dev
                    else:
                        train_metric_sums[k] = train_metric_sums[k] + v_dev
                train_counts += num_graphs

        # v30: ОДНА синхронизация GPU→CPU в конце эпохи
        train_loss_avg = (train_loss_sum / max(1, train_loss_count)).item()
        train_avg_metrics = {k: (v / max(1, train_counts)).item() for k, v in train_metric_sums.items()}

        # === Validation ===
        val_metrics = evaluate(model, val_loader, device, args, logger,
                              target_stats=target_stats, use_multi_gpu=use_multi_gpu)
        val_loss = val_metrics.get("loss", 0)
        elapsed = time.time() - t0

        # === Scheduler step (после вычисления val_loss) ===
        scheduler.step(val_loss)

        # === Early Stopping проверка ===
        metrics_to_track = {'val_loss': val_loss}
        if 'mu_mae' in val_metrics:
            metrics_to_track['val_mu_mae'] = val_metrics['mu_mae']
        if 'alpha_mae' in val_metrics:
            metrics_to_track['val_alpha_mae'] = val_metrics['alpha_mae']
        if 'gap_mae' in val_metrics:
            metrics_to_track['val_gap_mae'] = val_metrics['gap_mae']

        # v27: EarlyStopping вызов + расширенный лог
        stop = early_stopping(metrics_to_track, _underlying)

        # v28: текущий lr после scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        log_msg = (
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss_avg:.4f} | val_loss={val_loss:.4f} | "
            f"{' | '.join(f'{k}={v:.4f}' for k, v in val_metrics.items() if k != 'loss')} | "
            f"ES={early_stopping.format_counters()} | "
            f"lr={current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )
        # v27: RESET-метка (если что-то улучшилось)
        reset_str = early_stopping.format_resets()
        if reset_str:
            log_msg += f" | {reset_str}"
        # v28: метка лучшей эпохи (без сохранения на диск — только отметка)
        if early_stopping.last_saved:
            log_msg += " | ★ best"
        logger.info(log_msg)

        row = {
            "epoch": epoch,
            "train_loss": train_loss_avg,
            "val_loss": val_loss,
            "lr": current_lr,  # v28: lr в CSV
            "elapsed": elapsed,
        }
        for k, v in train_avg_metrics.items():
            row[f"train_{k}"] = v
        for k, v in val_metrics.items():
            if k != "loss":
                row[f"val_{k}"] = v
        history.append(row)

        if stop:
            logger.info(f"  → Early stopping на эпохе {epoch} (patience={args.patience})")
            break

    # Восстанавливаем лучшую модель (v27: в _underlying, не в обёртку DataParallel)
    early_stopping.restore_best_model(_underlying)

    # v28: сохранение best ckpt на диск ОДИН РАЗ — после restore, перед test
    torch.save(_underlying.state_dict(), ckpt_path)
    logger.info(f"→ Сохранён best checkpoint → {ckpt_path.name}")

    # === Test ===
    logger.info("\n=== Финальная оценка на test ===")
    test_metrics = evaluate(model, test_loader, device, args, logger,
                            prefix="test", target_stats=target_stats,
                            use_multi_gpu=use_multi_gpu)
    for k, v in test_metrics.items():
        logger.info(f"  test_{k}: {v:.4f}")

    # === Сохранение истории в CSV ===
    import csv

    # v28: дата+время в имени файла
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{args.output_dir}/history_{args.model}_{args.target}_{ts}.csv"  # v28: timestamp
    if history:
        keys = list(history[0].keys())
        # Добавляем test-метрики в последнюю строку
        for k, v in test_metrics.items():
            history[-1][f"test_{k}"] = v
        keys.extend([f"test_{k}" for k in test_metrics.keys()])
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)
        logger.info(f"История сохранена в {csv_path}")


def evaluate(model, loader, device, args, logger, prefix="val",
             target_stats: dict | None = None, use_multi_gpu: bool = False):
    """Оценка модели на лоадере.

    v29: если use_multi_gpu=True, loader возвращает list[Data],
    модель обёрнута в PyG DataParallel и принимает data_list.
    """
    from torch_geometric.data import Batch
    model.eval()
    # v32: унифицированный accumulator — раньше был AverageMeter для loss
    # + metric_sums dict для остальных метрик, дублирование логики.
    # Теперь всё в одном metric_sums, loss просто как ещё одна метрика.
    metric_sums = {}
    counts = 0

    with torch.no_grad():
        for batch_or_list in loader:
            if use_multi_gpu:
                data_list = batch_or_list
                batch = Batch.from_data_list(data_list).to(device)
                num_graphs = len(data_list)
                preds = model(data_list)
            else:
                batch = batch_or_list.to(device)
                num_graphs = batch.num_graphs
                preds = model(batch)
            # v32: noise_mode управляет, куда добавлять шум в evaluate()
            # test_only: только test (default, для robustness eval чистой модели)
            # train_val_test: train+val+test (consistent eval при обучении с шумом)
            # train_only: только train (evaluate без шума)
            # eval_only: val+test (no train augmentation, only eval-robustness)
            if args.noise > 0:
                if args.noise_mode == "test_only" and prefix == "test":
                    batch.pos = batch.pos + torch.randn_like(batch.pos) * args.noise
                elif args.noise_mode == "train_val_test" and prefix in ("val", "test"):
                    batch.pos = batch.pos + torch.randn_like(batch.pos) * args.noise
                elif args.noise_mode == "eval_only" and prefix in ("val", "test"):
                    batch.pos = batch.pos + torch.randn_like(batch.pos) * args.noise
                # train_only: no noise in evaluate()
            loss = compute_loss(preds, batch, args.target, target_stats)
            metrics = compute_metrics(preds, batch, args.target, target_stats)

            # v32: loss в тот же dict, что и остальные метрики
            metric_sums["loss"] = metric_sums.get("loss", 0.0) + loss.item() * num_graphs
            for k, v in metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v * num_graphs
            counts += num_graphs

    avg_metrics = {k: v / max(1, counts) for k, v in metric_sums.items()}
    return avg_metrics


if __name__ == "__main__":
    main()
