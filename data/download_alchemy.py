"""
Загрузка датасета Alchemy (полная версия v20191129).

Источник: https://alchemy.tencent.com/data/alchemy-v20191129.zip
Размер: ~136 МБ (zip), ~600 МБ после распаковки
Содержит:
  - Alchemy-v20191129/atom_9/...atom_12/  — SDF файлы молекул
  - Alchemy-v20191129/final_version.csv   — 12 квантово-механических свойств

Свойства в final_version.csv:
  - mu   (D, dipole moment)                      — наш главный таргет (часть A: скаляр)
  - alpha (a_0^3, Isotropic polarizability)      — наш второй таргет (часть A: скаляр)
  - gap  (Ha, LUMO-HOMO)                         — multi-task (скаляр)
  - HOMO, LUMO, U0, U, H, G, Cv, zpve, R2        — остальные свойства

Программа максимум (часть B):
  Вектор диполя μ (1×3) и тензор поляризуемости α (3×3) вычисляются отдельно
  через PySCF для подмножества молекул — см. src/dipole_pyscf.py
"""
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ALCHEMY_URL = "https://alchemy.tencent.com/data/alchemy-v20191129.zip"
DATA_DIR = Path(__file__).parent / "alchemy"


def download_alchemy(force: bool = False) -> None:
    """Скачать и распаковать датасет Alchemy (v20191129)."""
    # Проверяем, что данные уже есть
    csv_path = DATA_DIR / "Alchemy-v20191129" / "final_version.csv"
    if csv_path.exists() and not force:
        print(f"[OK] Alchemy уже скачан в {DATA_DIR}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "alchemy-v20191129.zip"

    print(f"[1/3] Скачиваю Alchemy (v20191129) из {ALCHEMY_URL}")
    print(f"      Размер ~136 МБ, может занять 1-3 минуты ...")
    urllib.request.urlretrieve(ALCHEMY_URL, zip_path)
    print(f"      Сохранено в {zip_path}")

    print(f"[2/3] Распаковываю ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()

    # Проверка структуры
    extracted = DATA_DIR / "Alchemy-v20191129"
    if not extracted.exists():
        # Поиск распакованной папки
        for d in DATA_DIR.iterdir():
            if d.is_dir() and (d / "final_version.csv").exists():
                extracted = d
                break

    print(f"[3/3] Готово. Данные в {extracted}")
    print(f"\nСтруктура:")
    for item in sorted(extracted.iterdir()):
        if item.is_dir():
            n_files = len(list(item.glob("*.sdf")))
            print(f"  {item.name}/  ({n_files} SDF файлов)")
        else:
            size = item.stat().st_size / 1e6
            print(f"  {item.name}  ({size:.1f} МБ)")


def inspect_csv() -> None:
    """Посмотреть структуру final_version.csv."""
    csv_path = DATA_DIR / "Alchemy-v20191129" / "final_version.csv"
    if not csv_path.exists():
        print(f"CSV не найден: {csv_path}")
        return

    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"\n=== final_version.csv ===")
    print(f"Размер: {df.shape[0]} молекул × {df.shape[1]} колонок")
    print(f"\nКолонки:")
    for i, c in enumerate(df.columns):
        print(f"  {i}: {c!r}")
    print(f"\nПервые 3 строки:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    download_alchemy(force="--force" in sys.argv)
    inspect_csv()
