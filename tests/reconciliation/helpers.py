"""Result-narrowing helpers for C8's tests.

Consumers of S5/S6 must narrow `Result[T]` before touching `.value`; these do
it once so the assertions stay readable and the suite type-checks under the
same mypy settings as the package.

Deliberately a local copy of the two lines `tests/pillars/synthetic.py` has for
C7, rather than an import of it: C8's unit tests should not depend on another
component's fixture module. `test_runner.py` does import that module, but only
where it drives the real `HamiltonEngine`.
"""

from __future__ import annotations

from seraph.shared_types import Err, Ok, Result

__all__ = ["as_err", "as_ok", "unwrap"]


def as_ok[T](res: Result[T]) -> Ok[T]:
    assert isinstance(res, Ok), f"expected Ok, got {res!r}"
    return res


def unwrap[T](res: Result[T]) -> T:
    return as_ok(res).value


def as_err[T](res: Result[T]) -> Err:
    assert isinstance(res, Err), f"expected Err, got {res!r}"
    return res
