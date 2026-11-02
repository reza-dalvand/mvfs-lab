from typing import Any, Dict, List

_PROJECTS: Dict[str, Any] = {}


def register_project(adapter: Any) -> None:
    """
    Register a project adapter.

    The adapter must expose at least:
    - name: str
    - display_name: str
    - data_contract: str
    - run(data_path=None, **kwargs) -> ProjectRunResult
    """
    name = getattr(adapter, "name", None)
    if not name:
        raise ValueError("Project adapter must define a non-empty 'name'.")

    _PROJECTS[name] = adapter


def get_project(name: str) -> Any:
    if name not in _PROJECTS:
        available = ", ".join(sorted(_PROJECTS.keys())) or "none"
        raise KeyError(f"Project '{name}' is not registered. Available projects: {available}")
    return _PROJECTS[name]


def list_projects() -> List[str]:
    return sorted(_PROJECTS.keys())