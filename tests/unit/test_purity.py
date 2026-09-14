from __future__ import annotations

import ast
from pathlib import Path

import pytest

PURE_MODULES = [
    "manifest",
    "windowing",
    "segmentation",
    "geometry",
    "findings",
    "resume",
]

FORBIDDEN_IMPORTS = {"aiohttp", "requests", "pydicom", "os", "pathlib"}

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "mammo_lejepa"


def _imported_top_level_modules(source: str) -> set[str]:
    tree = ast.parse(source)
    modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])

    return modules


@pytest.mark.parametrize("module_name", PURE_MODULES)
def test_pure_module_does_not_import_io_libraries(module_name: str) -> None:
    path = SRC_DIR / f"{module_name}.py"
    source = path.read_text(encoding="utf-8")

    imported = _imported_top_level_modules(source)
    forbidden_found = imported & FORBIDDEN_IMPORTS

    assert not forbidden_found, (
        f"{module_name}.py importa {forbidden_found}, prohibido para módulos "
        "puros (constitución, principio I)"
    )
