"""Packaging metadata checks: the PyPI-facing fields stay consistent.

These tests guard the release surface, not the runtime behaviour: the version in
``pyproject.toml`` has to match the one the package reports, the README has to be
the long description, and PEP 561's ``py.typed`` marker has to ship.

    python -m pytest tests/test_packaging.py
"""

from __future__ import annotations

import pathlib
from typing import Any

try:  # Python 3.11+ ships tomllib in the standard library
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

import jev_route

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def project_metadata() -> dict[str, Any]:
    """Return the ``[project]`` table of ``pyproject.toml``."""
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]


def test_declared_version_matches_the_package() -> None:
    """``pyproject.toml`` and ``jev_route.__version__`` must agree."""
    assert project_metadata()["version"] == jev_route.__version__


def test_pypi_facing_metadata_is_complete() -> None:
    """Every field PyPI renders is filled in."""
    metadata = project_metadata()

    assert metadata["name"] == "jev-route"
    assert 0 < len(metadata["description"]) <= 200
    assert metadata["readme"] == "README.md"
    assert metadata["requires-python"] == ">=3.10"
    assert metadata["license"] == "MIT"
    assert "LICENSE" in metadata["license-files"]
    assert metadata["keywords"]
    assert {"Homepage", "Repository", "Issues"} <= set(metadata["urls"])
    assert "Typing :: Typed" in metadata["classifiers"]
    assert any(
        classifier.startswith("Programming Language :: Python :: 3.1")
        for classifier in metadata["classifiers"]
    )


def test_type_marker_and_extras_ship_with_the_package() -> None:
    """PEP 561 typing marker plus the documented dependency extras exist."""
    metadata = project_metadata()

    assert (PROJECT_ROOT / "src" / "jev_route" / "py.typed").is_file()
    assert metadata["dependencies"] == ["httpx>=0.27", "pydantic>=2.5"]
    assert set(metadata["optional-dependencies"]) == {"fastapi", "dev"}
    assert "twine>=5.1" in metadata["optional-dependencies"]["dev"]
