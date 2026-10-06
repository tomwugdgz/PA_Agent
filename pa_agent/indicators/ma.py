"""Simple Moving Average (SMA / MA) — 简单移动平均。

与 :mod:`pa_agent.indicators.ema` 保持完全一致的约定：

- 输入 **oldest-first**（由调用方反转）；
- 输出同长度列表，预热期填 ``math.nan``；
- 第 ``period-1`` 位起为有效值。

与 EMA 的区别：MA 是窗口内算术平均，**没有递推**，
因此同一天算同一天的数据必然一致（EMA 依赖递推路径，
换一次增量/全量算法的起点不同会导致细微差异）。
"""
from __future__ import annotations

import math


def ma_full(values: list[float], period: int) -> list[float]:
    """计算简单移动平均。

    Args:
        values: 价格序列，**oldest first**。
        period: 均线周期（必须 >= 1）。

    Returns:
        与输入等长的列表：
        - 下标 ``0 .. period-2``为 ``nan``（预热）
        - 下标 ``period-1`` 起为有效均值

    Raises:
        ValueError: ``period`` < 1。
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    n = len(values)
    result = [math.nan] * n
    if n < period:
        return result

    # 滚动窗口和：先算第一个完整窗口，再逐位加减，
    # 时间复杂度 O(n)，不做 O(n*period) 的重复切片求和。
    window_sum = sum(values[:period])
    result[period - 1] = window_sum / period
    for i in range(period, n):
        window_sum += values[i] - values[i - period]
        result[i] = window_sum / period
    return result


def ma_series(
    values: list[float], periods: tuple[int, ...]
) -> dict[int, list[float]]:
    """一次算多条均线，避免重复遍历价格序列。

    Args:
        values: 价格序列，oldest first。
        periods: 均线周期元组，如 ``(5, 10, 20, 60)``。

    Returns:
        ``{period: 该周期的均线序列}``。
    """
    return {p: ma_full(values, p) for p in periods}
