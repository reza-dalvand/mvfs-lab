from mvfs_core.bootstrap import ensure_registered
from mvfs_core.registry import get_project, list_projects


def test_projects_are_registered():
    ensure_registered()

    names = list_projects()

    assert "uimufslr" in names
    assert "teruimufs" in names


def test_get_project_returns_adapter():
    ensure_registered()

    uim = get_project("uimufslr")
    teru = get_project("teruimufs")

    assert uim.name == "uimufslr"
    assert teru.name == "teruimufs"
    assert uim.data_contract == "uimufslr-original"
    assert teru.data_contract == "teruimufs-original"