from __future__ import annotations

import os
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

import httpx  # noqa: F401
import streamlit.runtime.scriptrunner.magic_funcs  # noqa: F401
import streamlit.web.bootstrap

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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

def _resolve_app_script() -> Path:
    candidates = [
        ROOT / "maimai_timing_align" / "app.py",
        ROOT / "src" / "maimai_timing_align" / "app.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("未找到 maimai_timing_align/app.py")

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    flag_options = {
        "server.port": 8503,
        "global.developmentMode": False,
        "server.maxUploadSize": 1000,
    }

    streamlit.web.bootstrap.load_config_options(flag_options=flag_options)
    flag_options["_is_running_with_streamlit"] = True
    streamlit.web.bootstrap.run(
        str(_resolve_app_script()),
        False,
        ['run'],
        flag_options,
    )