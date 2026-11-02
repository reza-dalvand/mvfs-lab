from .registry import list_projects, register_project


def ensure_registered() -> None:
    """
    Lazily import and register all known projects.

    Lazy import is intentional:
    it avoids importing heavy numerical code until actually needed.
    """
    names = set(list_projects())

    if "uimufslr" not in names:
        from uimufslr.project import UIMUFSLRProject

        register_project(UIMUFSLRProject())

    if "teruimufs" not in names:
        from teruimufs.project import TERUIMUFSProject

        register_project(TERUIMUFSProject())