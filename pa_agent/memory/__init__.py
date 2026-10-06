"""品种记忆库：按「品种 + 周期」隔离的历次分析归档、结算与画像。

与 :mod:`pa_agent.journal` 的分工：

* **journal**：按月分文件的**全库流水**，面向跨品种审计与全局统计。
* **memory**：按品种+周期隔离的**垂直档案**，面向"打开某个品种时
  立刻调出它自己的全部历史与准确率"。

两者都是 append-only 的纯外围模块，写盘失败绝不阻断主流程。
"""
from __future__ import annotations

from pa_agent.memory.settle import (
    accuracy_hint,
    rebuild_profile,
    settle_pending,
)
from pa_agent.memory.symbol_memory import (
    MEMORY_DIR,
    OutcomeRecord,
    SessionRecord,
    SymbolProfile,
    all_symbols,
    list_outcomes,
    list_sessions,
    load_profile,
    save_session,
    symbol_dir,
)

__all__ = [
    "MEMORY_DIR",
    "OutcomeRecord",
    "SessionRecord",
    "SymbolProfile",
    "accuracy_hint",
    "all_symbols",
    "list_outcomes",
    "list_sessions",
    "load_profile",
    "rebuild_profile",
    "save_session",
    "settle_pending",
    "symbol_dir",
]
