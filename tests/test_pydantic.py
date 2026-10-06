import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated

import numpy as np
import pytest
from pydantic import BaseModel, Field, RootModel, create_model

import thunk
from thunk._spec import ModelNode, compile_annotation


class Settings(BaseModel):
    name: str = Field(alias="label")
    count: int = Field(gt=0)
    shape: tuple[int, int]
    start: date


class Experiment(BaseModel):
    settings: Settings
    child: "Experiment | None" = None


@dataclass
class Wrapper:
    settings: Settings


def settings() -> Settings:
    return Settings(label="run", count=2, shape=(3, 4), start=date(2026, 1, 2))


def test_model_node() -> None:
    node = compile_annotation(Settings)
    assert isinstance(node, ModelNode)
    assert not node.contains_array
    node.validate(settings(), "config")
    assert node.describe()["schema"] == Settings.model_json_schema(mode="serialization")
    with pytest.raises(thunk.ValueTypeError, match="config.*Settings"):
        node.validate(settings().model_dump(), "config")


def test_roles_and_model_roundtrip(tmp_path: Path) -> None:
    calls = []

    def f(
        x: np.ndarray,
        config: "Settings",
        explicit: Annotated[Settings, thunk.Static()],
        many: list[Settings],
        optional: Settings | None,
        nested: Experiment,
        wrapped: Wrapper,
        mapping: dict[str, Settings],
        pair: tuple[Settings, ...],
        root: RootModel[list[int]],
    ) -> int:
        calls.append(config)
        return config.count

    config = settings()
    pfn = thunk.fn(f)
    inputs, opts = pfn.flatten(
        np.arange(3),
        config,
        config,
        [config],
        None,
        Experiment(settings=config, child=Experiment(settings=config)),
        Wrapper(config),
        {"first": config},
        (config,),
        RootModel[list[int]]([1, 2]),
    )
    assert list(inputs) == ["x"]
    assert list(opts) == [p.name for p in pfn._spec.static]
    assert opts["config"] is config
    path = tmp_path / "opts.json"
    pfn.save_opts(path, opts)
    restored = pfn.load_opts(path)
    assert not calls
    assert restored == opts
    assert type(restored["config"]) is Settings
    assert type(restored["config"].shape) is tuple
    assert type(restored["config"].start) is date
    assert type(restored["nested"].child) is Experiment
    assert type(restored["wrapped"]) is Wrapper
    assert type(restored["wrapped"].settings) is Settings
    assert type(restored["many"][0]) is Settings
    assert type(restored["mapping"]["first"]) is Settings
    assert type(restored["pair"]) is tuple
    assert pfn(inputs, restored) == 2
    doc = json.loads(path.read_text())
    assert doc["values"]["config"]["label"] == "run"


@pytest.mark.parametrize("raw", ["2", True, 2.5, 0])
def test_model_load_validates_fields_strictly(tmp_path: Path, raw: object) -> None:
    def f(config: Settings) -> None: ...

    pfn = thunk.fn(f)
    path = tmp_path / "opts.json"
    pfn.save_opts(path, {"config": settings()})
    doc = json.loads(path.read_text())
    doc["values"]["config"]["count"] = raw
    path.write_text(json.dumps(doc))
    with pytest.raises(thunk.StorageFormatError, match="count"):
        pfn.load_opts(path)


def test_model_save_requires_instance(tmp_path: Path) -> None:
    def f(config: Settings) -> None: ...

    path = tmp_path / "opts.json"
    with pytest.raises(thunk.ValueTypeError, match="Settings"):
        thunk.fn(f).save_opts(path, {"config": settings().model_dump()})
    assert not path.exists()


def test_model_fields_affect_fingerprint(tmp_path: Path) -> None:
    def f(config: BaseModel) -> None: ...

    original = create_model("Config", count=(int, ...))
    changed = create_model("Config", count=(str, ...))
    f.__annotations__["config"] = original
    path = tmp_path / "opts.json"
    thunk.fn(f).save_opts(path, {"config": original.model_validate({"count": 2})})
    f.__annotations__["config"] = changed
    with pytest.raises(thunk.SchemaMismatchError, match="config"):
        thunk.fn(f).load_opts(path)


def test_models_require_json_backend(tmp_path: Path) -> None:
    def explicit(config: Annotated[Settings, thunk.Data()]) -> None: ...
    def mixed(config: tuple[np.ndarray, Settings]) -> None: ...
    def output() -> Settings:
        raise AssertionError("body must not run")

    for f in (explicit, mixed):
        with pytest.raises(thunk.SpecError, match="custom Data"):
            thunk.fn(f)
    with pytest.raises(thunk.OutputCodecError, match="JSON storage"):
        thunk.fn(output).save_output(tmp_path / "out.h5", settings())


def test_models_can_be_skipped() -> None:
    def f(config: Annotated[Settings, thunk.Skip()]) -> None: ...

    assert thunk.fn(f).flatten(settings()) == ({}, {})
