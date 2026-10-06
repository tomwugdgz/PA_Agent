"""记忆库的结算与画像重算：让历史记录真正反哺下一次分析。

核心机制
--------
预测当时不知道结果，所以流程是**先存后结**：

1. :func:`save_session` 写入一条 ``sessions.jsonl``（``settled=False``）
2. 下次打开该品种时，:func:`settle_pending` 拉取后续 K 线，
   把上一条预测与实际走势对比，写入 ``outcomes.jsonl``
3. :func:`rebuild_profile` 重算该品种画像（命中率、均线形态分布、
   常见支撑阻力），供报告里展示"这个品种上次预测准不准"

结算的诚实边界
--------------
以下情况**不计入准确率**，只标记状态：

* ``no_bars``：拉不到后续 K 线
* ``insufficient``：后续 K 线不足 ``_SETTLE_LOOKAHEAD`` 的一半，
  走势还没走完，判定不可靠
* ``stale``：距上次分析超过 ``_SETTLE_MAX_AGE_MS``，行情已断档
* 实际方向判为 ``flat``：涨跌没超过 ``_DIRECTION_EPS_ATR`` × ATR，
  属于"没走出方向"，既不算对也不算���

宁可少算，不可算错——把"没方向"硬算成对或错会污染命中率，
让后续校准基于错误信号。
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from datetime import datetime
from typing import Any

from pa_agent.memory.symbol_memory import (
    _DIRECTION_EPS_ATR,
    _SETTLE_LOOKAHEAD,
    _SETTLE_MAX_AGE_MS,
    OutcomeRecord,
    SymbolProfile,
    _append_jsonl,
    _num,
    list_sessions,
    symbol_dir,
)

logger = logging.getLogger(__name__)


def settle_pending(
    *,
    symbol: str,
    timeframe: str,
    bars: list[Any],
    atr: float | None = None,
) -> list[OutcomeRecord]:
    """结算该品种尚未结算的会话，返回本次产生的结算记录。

    参数
    ----
    bars
        **newest-first** 的 K 线序列（与 KlineFrame.bars 同序）。
        通常就是当前 frame 的 bars——本K 线之后若还有更新的 K 线，
        上一条预测的结果就能判定。
    atr
        判定方向幅度用的 ATR。留 None 时退回用记录的 close/atr 字段。

    永不抛异常。
    """
    try:
        return _settle_impl(symbol, timeframe, bars, atr)
    except Exception as exc:  # noqa: BLE001
        logger.debug("记忆库结算失败（不影响报告）: %s", exc)
        return []


def _settle_impl(
    symbol: str, timeframe: str, bars: list[Any], atr: float | None
) -> list[OutcomeRecord]:
    # 找最后一条未结算的会话（只结算最近一条，避免批量错配）
    all_sessions = list_sessions(symbol, timeframe, limit=200)
    if not all_sessions:
        return []
    pending = [s for s in all_sessions if not s.get("settled")]
    if not pending:
        return []
    # all_sessions 已按时间倒序，第一条未结算的就是最近一次预测
    sess = pending[0]

    # 决策时刻用 **K 线时间**（bars_ts_ms），不是墙钟ts_ms。
    # 墙钟与行情时钟可能差时区（MT5 服务器实测偏 -3h），拿墙钟比对
    # K 线 ts_open 会得出"没有后续 K 线"的错误结论。
    # bars_ts_ms 缺失（老记录）才退回墙钟。
    open_ms = int(sess.get("bars_ts_ms") or sess.get("ts_ms") or 0)
    if not open_ms:
        return []
    # 陈旧性仍用墙钟——判断"是否过了很久没更新"只能看真实经过时间
    age = int(time.time() * 1000) - int(sess.get("ts_ms") or open_ms)
    if age > _SETTLE_MAX_AGE_MS:
        return [_write_outcome(sess, bars, atr, status="stale",
                               note=f"距上次分析 {age / 86400000:.1f} 天，行情已断档")]

    # bars 是 newest-first，需要找出比 open_ms 更新的那些（即已经走出来的）
    # bar.ts_open 是该根 K 线的开盘时间；> open_ms 表示在决策之后
    future = [b for b in bars if int(getattr(b, "ts_open", 0) or 0) > open_ms]
    future = sorted(future, key=lambda b: int(b.ts_open))  # oldest-first

    if not future:
        return [_write_outcome(sess, bars, atr, status="no_bars",
                               note="无决策之后的 K 线")]

    if len(future) < max(2, _SETTLE_LOOKAHEAD // 2):
        return [_write_outcome(sess, bars, atr, status="insufficient",
                               note=f"后续仅 {len(future)} 根 K 线，走势未走完")]

    window = future[:_SETTLE_LOOKAHEAD]
    high = max(float(b.high) for b in window)
    low = min(float(b.low) for b in window)
    settle_ms = int(window[-1].ts_open)

    entry_close = _num(sess.get("close"))
    ref_atr = _num(atr) or _num(sess.get("atr"))
    predicted = sess.get("direction")
    predicted_conf = _num(sess.get("direction_conf"))

    # ── 实际方向：按 ATR 幅度判定，避免噪声被当成方向 ──
    actual = "flat"
    move_atr = None
    if entry_close is not None and ref_atr and ref_atr > 0:
        up = (high - entry_close) / ref_atr
        dn = (entry_close - low) / ref_atr
        move_atr = round(max(up, dn) if up >= dn else dn, 3)
        if up >= _DIRECTION_EPS_ATR and up >= dn:
            actual = "bullish"
        elif dn >= _DIRECTION_EPS_ATR and dn > up:
            actual = "bearish"
        else:
            actual = "flat"

    # ── 方向命中：flat 不计入准确率（hit=None）──
    hit: bool | None = None
    if actual != "flat" and predicted in ("bullish", "bearish"):
        hit = (predicted == actual)

    # ── 多空计划触达判定 ──
    plans = sess.get("plans") or {}
    long_r = _plan_result(plans.get("long"), high, low)
    short_r = _plan_result(plans.get("short"), high, low)

    out = OutcomeRecord(
        session_id=str(sess.get("session_id") or ""),
        symbol=symbol,
        timeframe=timeframe,
        open_ts_ms=open_ms,
        settle_ts_ms=settle_ms,
        predicted=predicted,
        predicted_conf=predicted_conf,
        actual=actual,
        direction_hit=hit,
        high=high,
        low=low,
        move_atr=move_atr,
        long_result=long_r,
        short_result=short_r,
        status="ok",
        note="",
    )
    return [_write_outcome(sess, bars, atr, status="ok", prebuilt=out)]


def _plan_result(plan: Any, high: float, low: float) -> str:
    """判断一个价格计划后续是触达目标还是先打止损。

    简化口径：只要区间内**任一方向**同时覆盖了 entry 和 target，
    算 target_hit；若覆盖 entry 和 stop，算 stop_hit。
    因为 K 线数据没有严格逐根路径，这里只做粗判——
    宁可标neither 也不臆断顺序。
    """
    if not isinstance(plan, dict) or not plan.get("actionable"):
        return "no_plan"
    entry = _num(plan.get("entry"))
    target = _num(plan.get("target"))
    stop = _num(plan.get("stop"))
    if entry is None:
        return "no_plan"
    hit_target = target is not None and (low <= target <= high)
    hit_stop = stop is not None and (low <= stop <= high)
    if hit_target and hit_stop:
        # 同一根大波动可能同时覆盖——保守记neither，不假装知道先后
        return "neither"
    if hit_target:
        return "target_hit"
    if hit_stop:
        return "stop_hit"
    return "neither"


def _write_outcome(
    sess: dict[str, Any],
    bars: list[Any],
    atr: float | None,
    *,
    status: str,
    note: str = "",
    prebuilt: OutcomeRecord | None = None,
) -> OutcomeRecord:
    """写结算记录，并把对应 session 标记为已结算。"""
    if prebuilt is not None:
        out = prebuilt
    else:
        out = OutcomeRecord(
            session_id=str(sess.get("session_id") or ""),
            symbol=str(sess.get("symbol") or ""),
            timeframe=str(sess.get("timeframe") or ""),
            open_ts_ms=int(sess.get("ts_ms") or 0),
            settle_ts_ms=int(time.time() * 1000),
            predicted=sess.get("direction"),
            predicted_conf=_num(sess.get("direction_conf")),
            status=status,
            note=note,
        )

    d = symbol_dir(sess.get("symbol", ""), sess.get("timeframe", ""))
    _append_jsonl(d / "outcomes.jsonl", _outcome_dict(out))
    _mark_settled(d / "sessions.jsonl", out.session_id)
    return out


def _outcome_dict(o: OutcomeRecord) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(o)


def _mark_settled(sessions_path: Any, session_id: str) -> None:
    """把 sessions.jsonl 里对应那条的 ``settled`` 改为 True。

    append-only 语义下重写整个文件代价大，但 sessions.jsonl 单品种
    条数有限（每天几十条量级），且这是唯一能保持文件自洽的做法。
    失败只记 debug，不影响结算结果。
    """
    from pa_agent.memory.symbol_memory import MEMORY_DIR  # noqa: F401

    path = sessions_path
    try:
        if not path.is_file():
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        changed = False
        out_lines: list[str] = []
        for raw in lines:
            raw_s = raw.strip()
            if not raw_s:
                continue
            try:
                obj = json.loads(raw_s)
            except ValueError:
                out_lines.append(raw_s)
                continue
            if obj.get("session_id") == session_id and not obj.get("settled"):
                obj["settled"] = True
                out_lines.append(json.dumps(obj, ensure_ascii=False))
                changed = True
            else:
                out_lines.append(raw_s)
        if changed:
            path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.debug("标记已结算失败: %s", exc)


def rebuild_profile(symbol: str, timeframe: str) -> SymbolProfile:
    """重算并保存该品种画像（可随时调，随时可重算）。"""
    try:
        return _rebuild_impl(symbol, timeframe)
    except Exception as exc:  # noqa: BLE001
        logger.debug("画像重算失败: %s", exc)
        return SymbolProfile(symbol=symbol, timeframe=timeframe)


def _rebuild_impl(symbol: str, timeframe: str) -> SymbolProfile:
    from pa_agent.memory.symbol_memory import list_outcomes

    sessions = list_sessions(symbol, timeframe, limit=2000)
    outcomes = list_outcomes(symbol, timeframe, limit=2000)

    prof = SymbolProfile(
        symbol=symbol,
        timeframe=timeframe,
        updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        n_sessions=len(sessions),
        n_settled=len([o for o in outcomes if o.get("status") == "ok"]),
    )

    # ── 方向命中率（只统计有明确判定的）──
    hits = [o for o in outcomes if o.get("direction_hit") is not None]
    if hits:
        prof.direction_accuracy = round(
            sum(1 for o in hits if o["direction_hit"]) / len(hits), 4
        )

    # ── 方向分布 ──
    dcount: Counter[str] = Counter()
    for s in sessions:
        d = s.get("direction")
        if d in ("bullish", "bearish", "neutral"):
            dcount[d] += 1
    prof.direction_counts = dict(dcount)

    # ── 观望主导率：三分支里观望概率最高/ 或唯一可执行的占比 ──
    waits = 0
    for s in sessions:
        brs = s.get("branches") or []
        if not brs:
            continue
        top = max(brs, key=lambda b: b.get("probability") or 0.0)
        if top.get("key") == "wait":
            waits += 1
    if sessions:
        prof.wait_dominant_rate = round(waits / len(sessions), 4)

    # ── ATR 均值与价格区间 ──
    atrs = [s["atr"] for s in sessions if _num(s.get("atr"))]
    closes = [s["close"] for s in sessions if _num(s.get("close"))]
    if atrs:
        prof.atr_avg = round(sum(atrs) / len(atrs), 6)
    if closes:
        prof.price_min = round(min(closes), 6)
        prof.price_max = round(max(closes), 6)

    # ── 常见支撑阻力：取出现次数最多的 3 个 ──
    prof.common_supports = _top_levels([s.get("supports") for s in sessions])
    prof.common_resistances = _top_levels([s.get("resistances") for s in sessions])

    # ── 最近 10 次结算摘要 ──
    prof.recent_outcomes = [
        {
            "ts": o.get("open_ts_ms"),
            "predicted": o.get("predicted"),
            "actual": o.get("actual"),
            "hit": o.get("direction_hit"),
            "move_atr": o.get("move_atr"),
        }
        for o in outcomes[:10]
    ]

    d = symbol_dir(symbol, timeframe)
    d.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict

    (d / "profile.json").write_text(
        json.dumps(asdict(prof), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return prof


def _top_levels(levels_lists: list[Any], top_n: int = 3) -> list[float]:
    """从多次分析里挑出最常出现的支撑/阻力价位。

    按"四舍五入到 5 位小数"聚类——价格常有微小抖动，
    直接用浮点相等会一个都聚不到一起。
    """
    c: Counter[float] = Counter()
    for lv in levels_lists:
        for v in (lv or []):
            f = _num(v)
            if f is not None:
                c[round(f, 5)] += 1
    if not c:
        return []
    # 出现次数优先；次数相同时取更常被记录的（并列则取靠近现价难判定，取高者为先）
    ranked = sorted(c.items(), key=lambda kv: (-kv[1], -kv[0]))
    return [round(k, 5) for k, _ in ranked[:top_n]]


def accuracy_hint(symbol: str, timeframe: str) -> str:
    """给报告用的一行提示：上次这个品种预测准不准。"""
    prof = load_profile_safe(symbol, timeframe)
    if prof is None or not prof.n_settled:
        return "该品种暂无已结算的历史预测"
    if prof.direction_accuracy is None:
        return f"该品种已有 {prof.n_settled} 次结算，但方向未明确，不计命中率"
    return (
        f"该品种历史方向命中率 {prof.direction_accuracy:.0%}"
        f"（{prof.n_settled} 次有效结算/ 共 {prof.n_sessions} 次分析）"
    )


def load_profile_safe(symbol: str, timeframe: str) -> SymbolProfile | None:
    from pa_agent.memory.symbol_memory import load_profile

    try:
        p = load_profile(symbol, timeframe)
    except Exception:  # noqa: BLE001
        return None
    return p if p.n_sessions else None
