import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mvfs_core.bootstrap import ensure_registered
from mvfs_core.registry import get_project


def main():
    ensure_registered()

    project = get_project("uimufslr")

    # Optional:
    # python runs/run_uimufslr.py /path/to/dataset.mat
    data_path = sys.argv[1] if len(sys.argv) > 1 else None

    project.run(data_path=data_path)


if __name__ == "__main__":
    main()