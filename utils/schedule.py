# schedule.py

import math
from typing import List, Tuple

def from_exponential_schedule(schedule: List[Tuple[int, float]], step: int) -> float:
    schedule = sorted(schedule, key=lambda p: p[0])
    assert schedule[0][0] == 0
    for i in range(1, len(schedule)):
        if step < schedule[i][0]:
            s0, v0 = schedule[i - 1]
            s1, v1 = schedule[i]
            ratio = v0 / v1
            delta = step - s0
            duration = s1 - s0
            return v0 * math.exp(-math.log(ratio) * delta / duration)
    return schedule[-1][1]

def from_linear_schedule(schedule: List[Tuple[int, float]], step: int) -> float:
    schedule = sorted(schedule, key=lambda p: p[0])
    assert schedule[0][0] == 0
    for i in range(1, len(schedule)):
        if step < schedule[i][0]:
            s0, v0 = schedule[i - 1]
            s1, v1 = schedule[i]
            alpha = (step - s0) / (s1 - s0)
            return v0 + alpha * (v1 - v0)
    return schedule[-1][1]

def from_staircase_schedule(schedule: List[Tuple[int, float]], step: int) -> float:
    schedule = sorted(schedule, key=lambda p: p[0])
    assert schedule[0][0] == 0
    for s, v in reversed(schedule):
        if step >= s:
            return v
    return schedule[0][1]
