"""User-facing BEACON demo entry point.

This wrapper keeps the clearer ``demo_*`` naming while the implementation
remains in ``main_scholar.py``.
"""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    env_root = repo_root / "enviornment"
    for path in (here, repo_root, env_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    runpy.run_path(str(Path(__file__).with_name("main_scholar.py")), run_name="__main__")
