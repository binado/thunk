import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from helper_funcs import Population, explode, spectra

import thunk
from thunk import _opts_json

TESTS = Path(__file__).parent


def run_fresh(code: str) -> str:
    env = {**os.environ, "PYTHONPATH": str(TESTS)}
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_numpy_compilation_does_not_import_jax() -> None:
    run_fresh("""
        import sys
        import importlib.abc
        class NoJax(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "jax" or fullname.startswith("jax."):
                    raise AssertionError("thunk imported JAX")
        sys.meta_path.insert(0, NoJax())
        import numpy as np
        import thunk
        def f(x: np.ndarray) -> np.ndarray:
            raise AssertionError("body executed")
        thunk.fn(f)
        assert "jax" not in sys.modules
    """)


def test_restore_in_fresh_process_without_running_body(tmp_path: Path) -> None:
    pfn = thunk.fn(explode)
    x = np.linspace(0, 1, 5)
    pfn.save(tmp_path / "i.h5", tmp_path / "o.json", x, n=3)
    out = run_fresh(f"""
        import thunk
        from helper_funcs import explode
        pfn = thunk.fn(explode)
        args, kwargs = pfn.load(inputs={str(tmp_path / "i.h5")!r}, opts={str(tmp_path / "o.json")!r})
        print(len(args), sorted(kwargs), kwargs["n"], args[0].sum())
        try:
            pfn(*[pfn.load_inputs({str(tmp_path / "i.h5")!r}), pfn.load_opts({str(tmp_path / "o.json")!r})])
        except RuntimeError as e:
            print("body ran only when called")
    """)
    assert out.split("\n")[0] == f"1 ['n', 'tag'] 3 {x.sum()}"
    assert "body ran only when called" in out


def test_spectra_example_fresh_process(tmp_path: Path) -> None:
    pop = Population(np.ones((2, 3)), ["a", "b"], 2.0)
    seeds = np.arange(4.0)
    pfn = thunk.fn(spectra)
    pfn.save(tmp_path / "i.h5", tmp_path / "o.json", seeds, pop, bins=7)
    expected = spectra(seeds, pop, bins=7)
    pfn.save_output(tmp_path / "out.h5", expected)
    out = run_fresh(f"""
        import thunk
        from helper_funcs import spectra
        pfn = thunk.fn(spectra)
        args, kwargs = pfn.load(inputs={str(tmp_path / "i.h5")!r}, opts={str(tmp_path / "o.json")!r})
        r = spectra(*args, **kwargs)
        assert (pfn.load_output({str(tmp_path / "out.h5")!r}) == r).all()
        print(list(r))
    """)
    assert out.strip() == str(list(expected))


def test_save_leaves_both_destinations_untouched_on_encode_failure(
    tmp_path: Path,
) -> None:
    def f(x: np.ndarray, n: int) -> None: ...

    pfn = thunk.fn(f)
    i, o = tmp_path / "i.h5", tmp_path / "o.json"
    pfn.save(i, o, np.zeros(2), 1)
    before = (i.read_bytes(), o.read_bytes())
    with pytest.raises(thunk.ValueTypeError):
        pfn.save(i, o, np.ones(2), "bad")  # opts fail after inputs encoded
    assert (i.read_bytes(), o.read_bytes()) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["i.h5", "o.json"]


def test_save_failure_in_second_writer_cleans_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise OSError("disk full")

    def f(x: np.ndarray, n: int) -> None: ...

    pfn = thunk.fn(f)
    monkeypatch.setattr(_opts_json, "write_opts", boom)
    with pytest.raises(OSError):
        pfn.save(tmp_path / "i.h5", tmp_path / "o.json", np.zeros(2), 1)
    assert list(tmp_path.iterdir()) == []


def test_destinations_must_be_distinct(tmp_path: Path) -> None:
    def f(x: np.ndarray, n: int) -> None: ...

    p = tmp_path / "same"
    with pytest.raises(ValueError, match="distinct"):
        thunk.fn(f).save(p, tmp_path / "." / "same", np.zeros(1), 1)
    assert list(tmp_path.iterdir()) == []


def test_overwrite_replaces_and_new_file_has_normal_mode(tmp_path: Path) -> None:
    def f(n: int) -> None: ...

    pfn = thunk.fn(f)
    p = tmp_path / "o.json"
    pfn.save_opts(p, {"n": 1})
    assert p.stat().st_mode & 0o044  # readable beyond owner (umask permitting)
    pfn.save_opts(p, {"n": 2})
    assert pfn.load_opts(p) == {"n": 2}
    assert [q.name for q in tmp_path.iterdir()] == ["o.json"]
