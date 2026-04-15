from __future__ import annotations

import shutil
from pathlib import Path


"""
MLOps 架構自動重構腳本。

使用順序：
1. 先在專案根目錄執行本檔：
   python setup_project.py
2. 腳本會建立標準資料夾，並將核心模組移動到 src/。
3. 完成後再執行：
   python main_pipeline.py --n-trials 100

此腳本只負責建立架構與搬移核心 Python 模組，不會刪除舊資料夾。
"""


PROJECT_ROOT = Path(__file__).resolve().parent

DIRECTORIES_TO_CREATE = [
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "data" / "results",
    PROJECT_ROOT / "configs",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "outputs" / "figures",
    PROJECT_ROOT / "outputs" / "models",
]

CORE_MODULES = [
    "data_processor.py",
    "gpr_model_trainer.py",
    "optuna_optimizer.py",
]

KNOWN_MODULE_LOCATIONS = [
    PROJECT_ROOT / "scripts" / "sic_ml",
    PROJECT_ROOT,
]

IMPORT_REPLACEMENTS = {
    "from sic_ml.data_processor import DataProcessor": "from .data_processor import DataProcessor",
    "from sic_ml.gpr_model_trainer import GPRModelTrainer": "from .gpr_model_trainer import GPRModelTrainer",
    'data_path: str | Path = "DATA/cleaned_sic_sputtering_data.csv"': (
        'data_path: str | Path = "data/processed/cleaned_sic_sputtering_data.csv"'
    ),
    "PROJECT_ROOT = Path(__file__).resolve().parents[2]": "PROJECT_ROOT = Path(__file__).resolve().parents[1]",
    'output_dir = PROJECT_ROOT / "DATA"': 'output_dir = PROJECT_ROOT / "data" / "results"',
}


def create_directories() -> None:
    """建立標準 MLOps 目錄結構。"""

    print("[1/4] Creating MLOps directories...")
    for directory in DIRECTORIES_TO_CREATE:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  OK  {directory.relative_to(PROJECT_ROOT)}")


def ensure_src_package() -> None:
    """確保 src/ 是可 import 的 Python package。"""

    init_path = PROJECT_ROOT / "src" / "__init__.py"
    if init_path.exists() and init_path.read_text(encoding="utf-8").strip():
        print(f"[2/4] Existing src package kept: {init_path.relative_to(PROJECT_ROOT)}")
        return

    init_path.write_text(
        '''from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS_DIR = PROJECT_ROOT / ".codex_deps"

if LOCAL_DEPS_DIR.exists() and str(LOCAL_DEPS_DIR) not in sys.path:
    sys.path.append(str(LOCAL_DEPS_DIR))
''',
        encoding="utf-8",
    )
    print(f"[2/4] Created src package: {init_path.relative_to(PROJECT_ROOT)}")


def find_module(module_name: str) -> Path | None:
    """在已知位置尋找要搬移的核心模組。"""

    src_candidate = PROJECT_ROOT / "src" / module_name
    if src_candidate.exists():
        return src_candidate

    for base_dir in KNOWN_MODULE_LOCATIONS:
        candidate = base_dir / module_name
        if candidate.exists():
            return candidate

    matches = [
        path
        for path in PROJECT_ROOT.rglob(module_name)
        if ".git" not in path.parts
        and ".codex_deps" not in path.parts
        and "__pycache__" not in path.parts
    ]
    return matches[0] if matches else None


def move_core_modules() -> list[Path]:
    """將核心 ML 模組搬移到 src/，若已在 src/ 則略過。"""

    print("[3/4] Moving core modules into src/...")
    moved_or_existing: list[Path] = []
    src_dir = PROJECT_ROOT / "src"

    for module_name in CORE_MODULES:
        source_path = find_module(module_name)
        destination_path = src_dir / module_name

        if source_path is None:
            print(f"  WARN missing module: {module_name}")
            continue

        if source_path.resolve() == destination_path.resolve():
            print(f"  SKIP already in src: {destination_path.relative_to(PROJECT_ROOT)}")
            moved_or_existing.append(destination_path)
            continue

        if destination_path.exists():
            print(
                f"  SKIP destination exists: {destination_path.relative_to(PROJECT_ROOT)} "
                f"(source kept at {source_path.relative_to(PROJECT_ROOT)})"
            )
            moved_or_existing.append(destination_path)
            continue

        shutil.move(str(source_path), str(destination_path))
        print(f"  MOVE {source_path.relative_to(PROJECT_ROOT)} -> {destination_path.relative_to(PROJECT_ROOT)}")
        moved_or_existing.append(destination_path)

    return moved_or_existing


def normalize_imports(module_paths: list[Path]) -> None:
    """修正搬移到 src/ 後的內部 import。

    物理意義：
    - 重構前的模組位於 scripts/sic_ml/，因此使用 sic_ml.*。
    - 重構後的模組同屬 src package，應改為 relative import，讓 main_pipeline.py
      能透過 `from src.xxx import ...` 穩定載入。
    """

    print("[4/4] Normalizing imports for src package...")
    for module_path in module_paths:
        if not module_path.exists() or module_path.suffix != ".py":
            continue

        text = module_path.read_text(encoding="utf-8")
        updated = text
        for old_import, new_import in IMPORT_REPLACEMENTS.items():
            updated = updated.replace(old_import, new_import)

        if updated != text:
            module_path.write_text(updated, encoding="utf-8")
            print(f"  FIX  {module_path.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  OK   {module_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    print("Starting MLOps project refactoring...")
    print(f"Project root: {PROJECT_ROOT}")
    create_directories()
    ensure_src_package()
    module_paths = move_core_modules()
    normalize_imports(module_paths)
    print("Refactoring setup complete.")
    print("Next step: python main_pipeline.py --n-trials 100")


if __name__ == "__main__":
    main()
