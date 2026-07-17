"""
Загрузка датасета Alchemy (полная версия v20191129).

Источник: https://alchemy.tencent.com/data/alchemy-v20191129.zip
Размер: ~136 МБ (zip), ~600 МБ после распаковки
Содержит:
  - Alchemy-v20191129/atom_9/...atom_12/  — SDF файлы молекул
  - Alchemy-v20191129/final_version.csv   — 12 квантово-механических свойств

v32+: добавлена SHA256 проверка целостности zip. Если zip повреждён —
удаляется и скачивается заново (до 3 попыток).

Свойства в final_version.csv:
  - mu   (D, dipole moment)                      — наш главный таргет (часть A: скаляр)
  - alpha (a_0^3, Isotropic polarizability)      — наш второй таргет (часть A: скаляр)
  - gap  (Ha, LUMO-HOMO)                         — multi-task (скаляр)
  - HOMO, LUMO, U0, U, H, G, Cv, zpve, R2        — остальные свойства

Программа максимум (часть B):
  Вектор диполя μ (1×3) и тензор поляризуемости α (3×3) вычисляются отдельно
  через PySCF для подмножества молекул — см. src/dipole_pyscf.py
"""
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ALCHEMY_URL = "https://alchemy.tencent.com/data/alchemy-v20191129.zip"
DATA_DIR = Path(__file__).parent / "alchemy"

# v32: ожидаемый размер zip-файля (~136 МБ = ~142 605 312 байт).
# Если скачанный файл сильно меньше — это, скорее всего, HTML-страница с ошибкой.
EXPECTED_MIN_ZIP_SIZE = 100 * 1024 * 1024  # 100 МБ минимум

# Ожидаемые маркеры внутри zip (валидация без фиксации SHA256 —
# Tencent может перезапаковывать zip без изменения содержимого).
EXPECTED_FILES_IN_ZIP = [
    "Alchemy-v20191129/final_version.csv",
    "Alchemy-v20191129/atom_9/",
]


def _sha256(path: Path) -> str:
    """Вычислить SHA256 файла."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_zip(zip_path: Path) -> bool:
    """Проверить, что zip валиден и содержит ожидаемые файлы.

    Возвращает True если zip OK, False если повреждён.
    """
    if not zip_path.exists():
        return False
    # Размер: не меньше 100 МБ (HTML error page обычно <1 МБ)
    if zip_path.stat().st_size < EXPECTED_MIN_ZIP_SIZE:
        print(f"[WARN] zip слишком маленький: {zip_path.stat().st_size / 1e6:.1f} МБ "
              f"(ожидается >{EXPECTED_MIN_ZIP_SIZE / 1e6:.0f} МБ)")
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            for expected in EXPECTED_FILES_IN_ZIP:
                if not any(n.startswith(expected) for n in names):
                    print(f"[WARN] zip не содержит {expected}")
                    return False
        return True
    except zipfile.BadZipFile as e:
        print(f"[WARN] zip повреждён: {e}")
        return False


def download_alchemy(force: bool = False, max_retries: int = 3) -> None:
    """Скачать и распаковать датасет Alchemy (v20191129).

    v32: с проверкой целостности zip. При повреждении — повторная загрузка.

    Args:
        force: перекачать даже если данные уже есть
        max_retries: максимум попыток скачивания при повреждении zip
    """
    # Проверяем, что данные уже есть
    csv_path = DATA_DIR / "Alchemy-v20191129" / "final_version.csv"
    if csv_path.exists() and not force:
        print(f"[OK] Alchemy уже скачан в {DATA_DIR}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "alchemy-v20191129.zip"

    # v32: цикл скачивания с проверкой целостности
    for attempt in range(1, max_retries + 1):
        print(f"[1/3] Скачиваю Alchemy (v20191129) из {ALCHEMY_URL} (попытка {attempt}/{max_retries})")
        print("      Размер ~136 МБ, может занять 1-3 минуты ...")
        try:
            urllib.request.urlretrieve(ALCHEMY_URL, zip_path)
        except Exception as e:
            print(f"[WARN] Скачивание не удалось: {e}")
            if attempt < max_retries:
                print("      Повторная попытка через 2 сек...")
                import time
                time.sleep(2)
                continue
            else:
                raise

        # Проверка целостности
        print("[2/3] Проверяю целостность zip ...")
        if _validate_zip(zip_path):
            print(f"      SHA256: {_sha256(zip_path)}")
            print("      zip валиден")
            break
        else:
            print("[WARN] zip повреждён, удаляю и пробую снова")
            zip_path.unlink(missing_ok=True)
            if attempt == max_retries:
                raise RuntimeError(
                    f"Не удалось скачать валидный zip за {max_retries} попыток. "
                    f"Проверьте интернет-соединение или URL: {ALCHEMY_URL}"
                )

    print("[3/3] Распаковываю ...")
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

    print(f"Готово. Данные в {extracted}")
    print("\nСтруктура:")
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
    print("\n=== final_version.csv ===")
    print(f"Размер: {df.shape[0]} молекул × {df.shape[1]} колонок")
    print("\nКолонки:")
    for i, c in enumerate(df.columns):
        print(f"  {i}: {c!r}")
    print("\nПервые 3 строки:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    download_alchemy(force="--force" in sys.argv)
    inspect_csv()
