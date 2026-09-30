# -*- coding: utf-8 -*-
"""五层运行记录（journal）+ 贪婪调参器。

对外入口：
* ``layer_journal`` —— 数据 / 决策 / 风控 / 回测 / 执行 五层 JSONL 记录
* ``greedy_tuner`` —— 以确定性回测为评估器的贪婪坐标下降调参
"""
from pa_agent.journal.layer_journal import (  # noqa: F401
    LAYERS,
    JOURNAL_DIR,
    log_backtest,
    log_data,
    log_decision,
    log_execution,
    log_risk,
    log_tuning,
    record,
    stats_text,
    tail,
)

# journal 包刻意保持零 Qt、零网络依赖：任何层都能安全 import。
__all__ = [
    "LAYERS", "JOURNAL_DIR", "record", "stats_text", "tail",
    "log_data", "log_decision", "log_risk", "log_backtest", "log_execution",
    "log_tuning",
]
