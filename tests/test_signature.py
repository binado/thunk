import functools
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pytest
from helper_funcs import Population, spectra

import thunk


def test_roles_inferred() -> None:
    pfn = thunk.fn(spectra)
    spec = pfn._spec
    assert [p.name for p in spec.data] == ["seeds", "population"]
    assert [p.name for p in spec.static] == ["bins"]
    assert [p.name for p in spec.skip] == ["chunk_size"]


def test_flatten_fills_defaults_and_is_syntax_independent() -> None:
    def f(a: int, b: int = 2, *, c: int = 3) -> int:
        return a + b + c

    pfn = thunk.fn(f)
    assert pfn.flatten(1) == pfn.flatten(a=1) == pfn.flatten(1, 2, c=3)
    inputs, opts = pfn.flatten(1)
    assert inputs == {}
    assert list(opts) == ["a", "b", "c"]


def test_pos_only_and_kw_only_roundtrip_through_call() -> None:
    def f(a: int, /, b: int, *, c: int) -> tuple[int, int, int]:
        return (a, b, c)

    pfn = thunk.fn(f)
    assert pfn(*pfn.flatten(1, 2, c=3)) == f(1, 2, c=3)
    args, kwargs = pfn._spec.to_call({"a": 1, "b": 2, "c": 3})
    assert args == (1,)
    assert kwargs == {"b": 2, "c": 3}


def test_flatten_does_not_call_or_copy() -> None:
    calls = []

    def f(x: np.ndarray) -> None:
        calls.append(1)

    x = np.zeros(3)
    inputs, _ = thunk.fn(f).flatten(x)
    assert inputs["x"] is x
    assert not calls


def test_bad_call_raises_type_error() -> None:
    pfn = thunk.fn(lambda_free)
    with pytest.raises(TypeError):
        pfn.flatten()
    with pytest.raises(TypeError):
        pfn.flatten(1, 2, 3)


def lambda_free(a: int) -> int:
    return a


def test_var_args_rejected() -> None:
    def f(*args: int) -> None: ...
    def g(**kwargs: int) -> None: ...

    for func in (f, g):
        with pytest.raises(thunk.SpecError):
            thunk.fn(func)


def test_unannotated_persisted_param_rejected_but_skip_is_free() -> None:
    def bad(x) -> None: ...  # noqa: ANN001

    with pytest.raises(thunk.SpecError):
        thunk.fn(bad)

    class Opaque: ...

    def ok(x: int, o: Annotated[Opaque, thunk.Skip()]) -> None: ...

    thunk.fn(ok)

    def ok2(x: int, o: Annotated["Undefined", thunk.Skip()]) -> None: ...  # noqa: F821  # ty: ignore[unresolved-reference]

    thunk.fn(ok2)


def test_unresolvable_persisted_annotation_rejected() -> None:
    def f(x: "Undefined") -> None: ...  # noqa: F821  # ty: ignore[unresolved-reference]

    with pytest.raises(thunk.SpecError):
        thunk.fn(f)


def test_explicit_roles() -> None:
    def f(
        seed: Annotated[int, thunk.Data()],
        arr_ok: Annotated[list[int], thunk.Static()],
    ) -> None: ...

    spec = thunk.fn(f)._spec
    assert [p.name for p in spec.data] == ["seed"]
    assert [p.name for p in spec.static] == ["arr_ok"]


def test_static_with_arrays_rejected() -> None:
    def f(x: Annotated[np.ndarray, thunk.Static()]) -> None: ...

    with pytest.raises(thunk.SpecError, match="Static"):
        thunk.fn(f)


def test_two_roles_rejected() -> None:
    def f(x: Annotated[int, thunk.Data(), thunk.Skip()]) -> None: ...

    with pytest.raises(thunk.SpecError):
        thunk.fn(f)


def test_nested_marker_rejected() -> None:
    def f(x: list[Annotated[int, thunk.Skip()]]) -> None: ...

    with pytest.raises(thunk.SpecError, match="top-level"):
        thunk.fn(f)

    @dataclass
    class Bad:
        x: Annotated[int, thunk.Static()]

    def g(b: Bad) -> None: ...

    with pytest.raises(thunk.SpecError):
        thunk.fn(g)


def test_call_and_skipped_rules() -> None:
    pfn = thunk.fn(spectra)
    seeds = np.arange(3.0)
    pop = Population(np.ones(2), ["a"])
    inputs, opts = pfn.flatten(seeds, pop, bins=2)
    direct = spectra(seeds, pop, bins=2)
    assert np.array_equal(pfn(inputs, opts), direct)
    assert np.array_equal(
        pfn(inputs, opts, {"chunk_size": 1}), spectra(seeds, pop, bins=2, chunk_size=1)
    )
    with pytest.raises(TypeError, match="Skip"):
        pfn(inputs, opts, {"bins": 3})
    with pytest.raises(TypeError, match="unknown"):
        pfn({**inputs, "zzz": 1}, opts)
    with pytest.raises(TypeError, match="missing"):
        pfn({"seeds": seeds}, opts)


def test_required_skip_must_be_supplied() -> None:
    def f(x: int, y: Annotated[int, thunk.Skip()]) -> int:
        return x + y

    pfn = thunk.fn(f)
    inputs, opts = pfn.flatten(1, 2)
    with pytest.raises(TypeError, match="required skipped"):
        pfn(inputs, opts)
    assert pfn(inputs, opts, {"y": 5}) == 6


def test_param_named_like_internals() -> None:
    def f(inputs: int, opts: int, skipped: int, self: int = 0) -> int:
        return inputs + opts + skipped + self

    pfn = thunk.fn(f)
    assert pfn(*pfn.flatten(inputs=1, opts=2, skipped=3)) == 6


def test_callable_instance_and_bound_method() -> None:
    class Scale:
        def __init__(self, k: int) -> None:
            self.k = k

        def __call__(self, x: int) -> int:
            return x * self.k

        def method(self, x: int) -> int:
            return x + self.k

    for target in (Scale(3), Scale(3).method):
        pfn = thunk.fn(target)
        assert pfn(*pfn.flatten(2)) == target(2)


def test_partial_resolves_postponed_annotations_from_wrapped_function() -> None:
    def f(a: int, p: "Population") -> int:
        return a

    f.__globals__["Population"] = Population
    pfn = thunk.fn(functools.partial(f, 1))
    assert [p.name for p in pfn._spec.params] == ["p"]
