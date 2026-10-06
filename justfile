default:
    @just --list

setup:
    uv sync --all-groups

lock:
    uv lock

lint:
    uv run ruff check .

fmt:
    uv run ruff check --fix .
    uv run ruff format .

fmt-check:
    uv run ruff format --check .

typecheck:
    uv run ty check

test *args:
    uv run pytest {{args}}

hooks:
    uv run prek run --all-files

build:
    uv build

check: lint fmt-check typecheck test build
    uv lock --check
