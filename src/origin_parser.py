from __future__ import annotations

import importlib
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OriginExtractionResult:
    """單一 .opju 檔案的抽取結果。"""

    opju_path: Path
    status: str
    exported_files: tuple[Path, ...] = ()
    message: str = ""


class OriginExtractor:
    """Origin Project (.opju) 自動抽取器。

    核心工作：
    1. 背景開啟 `.opju`。
    2. 掃描所有可能的 worksheet。
    3. 尋找包含 Voltage/Current 欄位的表格。
    4. 匯出為 CSV，檔名加上 `_extracted_from_origin.csv`。
    5. 安全關閉 Origin COM session，避免背景 Origin.exe 殘留。

    自動安裝邏輯：
    - `_load_originpro()` 會先嘗試 `import originpro`。
    - 若 ImportError，且 `auto_install=True`，會在程式內部執行：
      `python -m pip install originpro`
    - 安裝完成後重新 import。

    注意：
    - `originpro` 是 OriginLab 官方 Python automation package。
    - 真正開啟 `.opju` 仍需要 Windows 上已安裝 Origin 且 COM automation 可用。
    - 若本機沒有 Origin，ETL 會回報 skipped/error，不應中斷 Excel/CSV 清洗主流程。
    """

    def __init__(
        self,
        root_dir: str | Path,
        auto_install: bool = True,
        visible: bool = False,
        overwrite: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.auto_install = auto_install
        self.visible = visible
        self.overwrite = overwrite
        self._origin_module: Any | None = None

    def _load_originpro(self):
        """動態載入 originpro；缺套件時自動 pip install。

        這是本模組的免手動配置關鍵。為了避免 import 本檔時就安裝套件，
        subprocess install 只在真正需要處理 `.opju` 時才會執行。
        """

        if self._origin_module is not None:
            return self._origin_module

        try:
            self._origin_module = importlib.import_module("originpro")
            return self._origin_module
        except ImportError:
            if not self.auto_install:
                raise

        logger.info("originpro is not installed. Installing with pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "originpro"])
        self._origin_module = importlib.import_module("originpro")
        return self._origin_module

    def extract_all(self) -> list[OriginExtractionResult]:
        """掃描 root_dir 之下所有 .opju 並匯出可用 worksheet。"""

        opju_files = sorted(self.root_dir.rglob("*.opju"))
        if not opju_files:
            return []

        results: list[OriginExtractionResult] = []
        for opju_path in opju_files:
            results.append(self.extract_project(opju_path))
        return results

    def extract_project(self, opju_path: str | Path) -> OriginExtractionResult:
        """開啟單一 Origin project 並匯出 I-V worksheets。"""

        opju_path = Path(opju_path)
        try:
            origin = self._load_originpro()
        except Exception as exc:  # noqa: BLE001
            return OriginExtractionResult(opju_path, "error", message=f"Cannot load originpro: {exc}")

        exported: list[Path] = []
        try:
            self._open_origin_project(origin, opju_path)
            worksheets = self._iter_candidate_worksheets(origin)

            for index, worksheet in enumerate(worksheets, start=1):
                frame = self._worksheet_to_dataframe(worksheet)
                if frame.empty or not self._looks_like_iv_table(frame):
                    continue

                output_path = self._build_output_path(opju_path, worksheet, index)
                if output_path.exists() and not self.overwrite:
                    exported.append(output_path)
                    continue

                frame.to_csv(output_path, index=False, encoding="utf-8-sig")
                exported.append(output_path)

            status = "extracted" if exported else "skipped"
            message = "" if exported else "No worksheet with Voltage/Current columns found"
            return OriginExtractionResult(opju_path, status, tuple(exported), message)
        except Exception as exc:  # noqa: BLE001
            return OriginExtractionResult(opju_path, "error", tuple(exported), str(exc))
        finally:
            self._close_origin(origin)

    def _open_origin_project(self, origin, opju_path: Path) -> None:
        """透過 Origin COM automation 在背景開啟 project。

        COM 操作說明：
        - `originpro` 會在 Windows 背景啟動 Origin instance。
        - `set_show(False)` 可避免 GUI 跳出干擾自動化。
        - `new()` 清空目前 session，降低跨 project worksheet 殘留風險。
        - `open()` 將 .opju 載入目前 Origin session。
        """

        if hasattr(origin, "set_show"):
            origin.set_show(self.visible)
        if hasattr(origin, "new"):
            origin.new()
        if not hasattr(origin, "open"):
            raise RuntimeError("originpro module does not expose open().")

        opened = origin.open(str(opju_path))
        if opened is False:
            raise RuntimeError(f"Origin failed to open project: {opju_path}")

    def _iter_candidate_worksheets(self, origin) -> list[Any]:
        """盡可能以多種 originpro API 掃描 worksheets。

        originpro 在不同版本的 page/sheet API 名稱可能略有差異，因此這裡採
        防禦式 discovery：
        - 優先嘗試 pages/layers。
        - 再嘗試 find_sheet。
        - 只收集具備 `to_df()` 的 worksheet-like 物件。
        """

        worksheets: list[Any] = []

        if hasattr(origin, "pages"):
            for call_args in ((), ("w",)):
                try:
                    pages = origin.pages(*call_args)
                except Exception:
                    continue
                for page in list(pages or []):
                    if hasattr(page, "to_df"):
                        worksheets.append(page)
                    layers = getattr(page, "layers", None)
                    if layers is not None:
                        try:
                            for layer in layers:
                                if hasattr(layer, "to_df"):
                                    worksheets.append(layer)
                        except Exception:
                            pass

        if not worksheets and hasattr(origin, "find_sheet"):
            for index in range(1, 500):
                worksheet = None
                for args in (("w", index), ("w", str(index)), (index,)):
                    try:
                        worksheet = origin.find_sheet(*args)
                    except Exception:
                        worksheet = None
                    if worksheet is not None:
                        break
                if worksheet is None:
                    if index > 20:
                        break
                    continue
                if hasattr(worksheet, "to_df"):
                    worksheets.append(worksheet)

        # 去除同一 COM object 被不同 discovery path 重複收集的情況。
        unique: list[Any] = []
        seen: set[int] = set()
        for worksheet in worksheets:
            key = id(worksheet)
            if key not in seen:
                seen.add(key)
                unique.append(worksheet)
        return unique

    @staticmethod
    def _worksheet_to_dataframe(worksheet) -> pd.DataFrame:
        """將 Origin worksheet 轉為 pandas DataFrame。"""

        if not hasattr(worksheet, "to_df"):
            return pd.DataFrame()
        frame = worksheet.to_df()
        if frame is None:
            return pd.DataFrame()
        frame = pd.DataFrame(frame)
        frame.columns = [str(column).strip() for column in frame.columns]
        return frame

    @staticmethod
    def _looks_like_iv_table(frame: pd.DataFrame) -> bool:
        """判斷 worksheet 是否包含 I-V curve 所需的 Voltage/Current 欄位。"""

        compact_columns = [re.sub(r"[^a-z0-9]+", "", str(column).lower()) for column in frame.columns]
        has_voltage = any("voltage" in column or column in {"v", "ch1voltage"} for column in compact_columns)
        has_current = any("current" in column or column in {"i", "ch1current"} for column in compact_columns)
        return has_voltage and has_current

    def _build_output_path(self, opju_path: Path, worksheet, index: int) -> Path:
        worksheet_name = getattr(worksheet, "name", None) or getattr(worksheet, "lname", None) or f"sheet_{index}"
        safe_sheet = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(worksheet_name)).strip("_") or f"sheet_{index}"
        return opju_path.with_name(f"{opju_path.stem}_{safe_sheet}_extracted_from_origin.csv")

    @staticmethod
    def _close_origin(origin) -> None:
        """安全釋放 Origin COM session。

        COM 物件若未關閉，可能殘留 Origin.exe 並持續占用記憶體或鎖住 .opju。
        `exit()` 是 originpro 常見釋放方式；若版本不支援，退而求其次執行 LabTalk
        清 project 指令。
        """

        try:
            if hasattr(origin, "exit"):
                origin.exit()
                return
        except Exception:
            pass

        try:
            if hasattr(origin, "lt_exec"):
                origin.lt_exec("doc -s; doc -n;")
        except Exception:
            pass
