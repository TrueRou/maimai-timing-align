from __future__ import annotations

import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _patch_streamlit_metadata_for_frozen() -> None:
    if not getattr(sys, "frozen", False):
        return

    original_version = importlib_metadata.version

    def _version(name: str) -> str:
        try:
            return original_version(name)
        except importlib_metadata.PackageNotFoundError:
            if name == "streamlit":
                # PyInstaller onefile 在某些平台不会带 dist-info 元数据。
                return "0.0.0"
            raise

    importlib_metadata.version = _version  # type: ignore[assignment]


_patch_streamlit_metadata_for_frozen()

from maimai_timing_align.app import run_app  # noqa: E402

if __name__ == "__main__":
    run_app()
