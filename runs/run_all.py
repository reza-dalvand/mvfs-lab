import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mvfs_core.bootstrap import ensure_registered
from mvfs_core.registry import get_project, list_projects


def main():
    ensure_registered()

    # You can pass project names manually:
    # python runs/run_all.py uimufslr teruimufs
    names = sys.argv[1:] if len(sys.argv) > 1 else list_projects()

    for name in names:
        print("\n")
        print("#" * 80)
        print(f"# Running project: {name}")
        print("#" * 80)

        project = get_project(name)
        project.run()


if __name__ == "__main__":
    main()