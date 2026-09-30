# -*- coding: utf-8 -*-
"""五层运行记录：数据 / 决策 / 风控 / 回测 / 执行。

设计原则（用户口径：底层尽量不动）
----------------------------------
* **纯外围**：只提供 append-only 的 JSONL 落盘，不参与任何业务判断；
  各层在既有流程的收尾处调一次 ``log_*`` 即可，主流程失败不受影响
  （所有写盘异常都吞掉只记 debug 日志）。
* **一层一文件**：``JOURNAL_DIR / YYYY-MM / <layer>.jsonl``，
  按月分目录自然归档，每行一个 JSON 事件，可直接用 pandas / jq 分析。
* **可追溯**：每行都带 ``ts_ms``（毫秒时间戳）、``ts``（本地时间）、
  ``layer``、``kind``、``symbol``、``timeframe``，跨层可以用
  symbol+timeframe+时间窗对齐（例如决策层事件 ↔ 执行层订单）。

层与事件的对应关系
------------------
====  ==========================  ====================================
层级  记录时机                    典型 kind
====  ==========================  ==========================
data  每次报告生成时的数据快照    report_input
decision  报告产出时记 AI 决策    laya_report
risk  报告产出时记风险告警        plan_warnings / low_confidence
backtest  每次回测/调参评估后     backtest_run / tune_eval
execution  MT5 下单返回后          order_ok / order_rejected / order_error
tuning  贪婪调参结束后             tune_summary
====  ==========================  ====================================
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pa_agent.config.paths import EXPERIENCE_DIR

logger = logging.getLogger(__name__)

#: 记录根目录（挂在 experience 下，与经验库同属"学习资产"）
JOURNAL_DIR: Path = EXPERIENCE_DIR / "journal"

#: 五层 + tuning（调参总结；调参的每次评估落在 backtest 层）
LAYERS: tuple[str, ...] = (
    "data", "decision", "risk", "backtest", "execution", "tuning",
)


def _layer_dir(layer: str) -> Path:
    """某层当月的 JSONL 文件路径（按月分目录）。"""
    month = datetime.now().strftime("%Y-%m")
    d = JOURNAL_DIR / month
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{layer}.jsonl"


def record(
    layer: str,
    kind: str,
    payload: dict[str, Any],
    *,
    symbol: str = "",
    timeframe: str = "",
) -> bool:
    """向指定层追加一条事件。返回是否成功（失败只记日志，绝不抛出）。

    所有层的公共字段：``ts_ms`` / ``ts`` / ``layer`` / ``kind`` /
    ``symbol`` / ``timeframe``；``payload`` 里的业务字段平铺在行内。
    """
    if layer not in LAYERS:
        logger.debug("journal: 未知层 %r，事件丢弃", layer)
        return False
    try:
        now_ms = int(time.time() * 1000)
        row: dict[str, Any] = {
            "ts_ms": now_ms,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "layer": layer,
            "kind": kind,
            "symbol": symbol,
            "timeframe": timeframe,
        }
        row.update(payload or {})
        path = _layer_dir(layer)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return True
    except Exception as exc:  # noqa: BLE001 - 记录绝不影响主流程
        logger.debug("journal 写入失败（layer=%s kind=%s）: %s", layer, kind, exc)
        return False


# ── 各层便捷入口（参数即文档） ───────────────────────────────────────────────


def log_data(
    *, symbol: str = "", timeframe: str = "", n_bars: int | None = None,
    close: float | None = None, atr: float | None = None,
    device: str = "", latency_ms: float | None = None,
    n_errors: int = 0, extra: dict[str, Any] | None = None,
) -> bool:
    """数据层：报告生成时的输入快照（K 线数量、ATR、推理设备等）。"""
    payload: dict[str, Any] = {"n_bars": n_bars, "close": close, "atr": atr,
                               "device": device, "latency_ms": latency_ms,
                               "n_errors": n_errors}
    if extra:
        payload.update(extra)
    return record("data", "report_input", payload,
                  symbol=symbol, timeframe=timeframe)


def log_decision(
    *, symbol: str = "", timeframe: str = "", source: str = "laya",
    answers: dict[str, Any] | None = None,
    plans: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> bool:
    """决策层：AI 每次分析的结论（答题摘要 + 双向价格计划摘要）。"""
    payload: dict[str, Any] = {"source": source,
                               "answers": answers or {},
                               "plans": plans or {}}
    if extra:
        payload.update(extra)
    return record("decision", f"{source}_report", payload,
                  symbol=symbol, timeframe=timeframe)


def log_risk(
    *, symbol: str = "", timeframe: str = "",
    warnings: list[str] | None = None,
    low_confidence: bool = False,
    max_rr: float | None = None,
    extra: dict[str, Any] | None = None,
) -> bool:
    """风控层：报告里的风险告警 / 低置信度 / 超常盈亏比。"""
    payload: dict[str, Any] = {"warnings": warnings or [],
                               "low_confidence": low_confidence,
                               "max_rr": max_rr}
    if extra:
        payload.update(extra)
    kind = "low_confidence" if low_confidence else "plan_warnings"
    return record("risk", kind, payload, symbol=symbol, timeframe=timeframe)


def log_backtest(
    *, symbol: str = "", timeframe: str = "", kind: str = "backtest_run",
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> bool:
    """回测层：每次确定性回测（含贪婪调参的每次试评估 tune_eval）。"""
    payload: dict[str, Any] = {"params": params or {}, "metrics": metrics or {}}
    if extra:
        payload.update(extra)
    return record("backtest", kind, payload, symbol=symbol, timeframe=timeframe)


def log_execution(
    *, symbol: str = "", timeframe: str = "",
    ok: bool, retcode: int | None = None,
    direction: str = "", order_kind: str = "",
    entry: float | None = None, stop_loss: float | None = None,
    take_profit: float | None = None, lot: float | None = None,
    ticket: int | None = None, message: str = "",
) -> bool:
    """执行层：MT5 下单结果（成功/被拒/异常都记，含 retcode 与票号）。"""
    return record(
        "execution",
        "order_ok" if ok else "order_rejected",
        {"ok": ok, "retcode": retcode, "direction": direction,
         "order_kind": order_kind, "entry": entry, "stop_loss": stop_loss,
         "take_profit": take_profit, "lot": lot, "ticket": ticket,
         "message": message},
        symbol=symbol, timeframe=timeframe,
    )


def log_tuning(
    *, symbol: str = "", timeframe: str = "",
    baseline: dict[str, Any] | None = None,
    best: dict[str, Any] | None = None,
    accepted: list[dict[str, Any]] | None = None,
    n_evals: int = 0,
) -> bool:
    """tuning 层：贪婪调参一轮的总结（起点 / 最优 / 接受的移动）。"""
    return record(
        "tuning", "tune_summary",
        {"baseline": baseline or {}, "best": best or {},
         "accepted": accepted or [], "n_evals": n_evals},
        symbol=symbol, timeframe=timeframe,
    )


# ── 读侧：统计与最近事件（GUI 展示用） ──────────────────────────────────────


def tail(layer: str, n: int = 10) -> list[dict[str, Any]]:
    """读某层最近 n 条事件（跨当月与上月文件，按 ts_ms 倒序）。失败返回 []。"""
    try:
        rows: list[dict[str, Any]] = []
        months = sorted({p.parent.name for p in JOURNAL_DIR.glob(f"*/{layer}.jsonl")},
                        reverse=True)[:2]
        for month in months:
            path = JOURNAL_DIR / month / f"{layer}.jsonl"
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        rows.sort(key=lambda r: r.get("ts_ms", 0), reverse=True)
        return rows[:n]
    except Exception as exc:  # noqa: BLE001
        logger.debug("journal tail 失败（layer=%s）: %s", layer, exc)
        return []


def stats_text(days: int = 7) -> str:
    """近 N 天各层事件数的中文摘要（GUI「运行记录」用）。"""
    try:
        months = sorted({p.parent.name for p in JOURNAL_DIR.glob("*/*.jsonl")},
                        reverse=True)[:2]
        counts: dict[str, int] = {layer: 0 for layer in LAYERS}
        last_ts: dict[str, str] = {}
        cutoff_ms = int((time.time() - days * 86400) * 1000)
        for month in months:
            for layer in LAYERS:
                path = JOURNAL_DIR / month / f"{layer}.jsonl"
                if not path.is_file():
                    continue
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("ts_ms", 0) >= cutoff_ms:
                        counts[layer] += 1
                        if layer not in last_ts or row.get("ts_ms", 0) > 0:
                            last_ts[layer] = str(row.get("ts", ""))
        lines = [f"近 {days} 天运行记录（experience/journal/）："]
        names = {"data": "数据层", "decision": "决策层", "risk": "风控层",
                 "backtest": "回测层", "execution": "执行层", "tuning": "调参层"}
        for layer in LAYERS:
            mark = f"（最近 {last_ts[layer]}）" if counts[layer] and layer in last_ts else ""
            lines.append(f"· {names[layer]}：{counts[layer]} 条{mark}")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"统计失败：{exc}"
