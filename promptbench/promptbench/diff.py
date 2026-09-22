from __future__ import annotations

from pydantic import BaseModel

from common.results import EvalResult, RunResult


class RunDiff(BaseModel):
    regressions: list[EvalResult]
    improvements: list[EvalResult]


def compare_runs(old: RunResult, new: RunResult) -> RunDiff:
    old_by_key = {(r.case_id, r.model): r for r in old.results}
    regressions: list[EvalResult] = []
    improvements: list[EvalResult] = []

    for new_result in new.results:
        key = (new_result.case_id, new_result.model)
        old_result = old_by_key.get(key)
        if old_result is None:
            continue
        if old_result.passed and not new_result.passed:
            regressions.append(new_result)
        elif not old_result.passed and new_result.passed:
            improvements.append(new_result)

    return RunDiff(regressions=regressions, improvements=improvements)
