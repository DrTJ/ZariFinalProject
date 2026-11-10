"""Cross-platform entry point for Mac and Windows.

Mac uses the vendored ``.venv`` (Python 3.11, Apple Silicon).
Windows uses ``.venv-windows``, created by ``setup_windows.ps1`` and left
separate so the Mac environment is never replaced.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def runtime_python() -> Path:
    if sys.platform == "win32":
        return ROOT / ".venv-windows" / "Scripts" / "python.exe"
    return ROOT / ".venv" / "bin" / "python"


def main() -> None:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("TORCH_HOME", str(ROOT / "models"))
    os.chdir(ROOT)

    py = runtime_python()
    if not py.is_file():
        if sys.platform == "win32":
            raise SystemExit(
                "Windows runtime is missing. From this folder run setup_windows.ps1 "
                "(or setup_windows.bat). That creates .venv-windows and does not touch .venv."
            )
        raise SystemExit(f"Mac runtime was not found at {py}")

    if Path(sys.executable).resolve() != py.resolve():
        os.execv(str(py), [str(py), str(Path(__file__).resolve()), *sys.argv[1:]])

    sys.path.insert(0, str(ROOT))
    from cascade_mot.run import main as run_main

    run_main()


if __name__ == "__main__":
    main()
