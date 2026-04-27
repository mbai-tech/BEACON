"""User-facing Bug2 demo entry point."""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    for path in (here, repo_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    runpy.run_path(str(Path(__file__).with_name("main_bug2.py")), run_name="__main__")
