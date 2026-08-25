"""Smoke tests: the package imports and core dependencies are usable."""

import importlib

import risk_scoring


def test_package_imports():
    assert risk_scoring.__version__


def test_core_dependencies_importable():
    for module in [
        "fastapi",
        "uvicorn",
        "pandas",
        "numpy",
        "sklearn",
        "mlflow",
        "sqlalchemy",
        "psycopg",
        "pydantic",
        "httpx",
    ]:
        importlib.import_module(module)
