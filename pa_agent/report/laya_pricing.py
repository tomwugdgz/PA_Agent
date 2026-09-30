# -*- coding: utf-8 -*-
"""确定性定价：结构优先 + ATR 兜底（用户选定的「混合口径」）。

原则
----
价格**永远由本地代码**算，Laya 只提供「该不该做、往哪个方向做」的概率。
这样报告里的每个数字都可复现、可回测，不依赖任何生成式输出。

规则（做多方向，做空镜像）
--------------------------
1. 挂单价   = 最近支撑 + entry_offset_atr × ATR
             （挂回调买单：略高于支撑，避免差一跳成交不了）
2. 止损     = 做多结构失效位（invalidation_long，即 supports[0]）
             − stop_buffer_atr × ATR
3. 目标     = measured_move 投影（kind 含 up 的第一条）
             ；无 MM 时 = entry + fallback_target_r × R
             （R = entry − stop）
4. 任一结构位缺失 → 该项回落到纯 ATR 倍数，并在结果里标 `fallback=True`
5. 结构位完全缺失 → 方向直接标「无有效结构」，不出价格
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PricePlan:
    """一个方向的可执行价格计划。"""

    direction: str                    # long | short | none
    actionable: bool                  # False = 不给价格（结构缺失/方向不可靠）
    reason: str = ""                  # 不可执行时的中文原因
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    risk_per_unit: float | None = None     # R = |entry - stop|
    rr_ratio: float | None = None          # 目标 / 风险
    entry_fallback: bool = False           # 挂单价用了 ATR 兜底
    stop_fallback: bool = False            # 止损用了 ATR 兜底
    target_fallback: bool = False          # 目标用了 R 倍数兜底
    notes: tuple[str, ...] = field(default_factory=tuple)


def _round_price(value: float, tick: float | None) -> float:
    """按最小变动价位取整；无 tick 信息时保留 5 位小数。"""
    if tick and tick > 0:
        steps = round(value / tick)
        return round(steps * tick, 10)
    return round(value, 5)


def _pick_mm_target(
    moves: tuple[Any, ...], direction: str
) -> tuple[float | None, str]:
    """挑一个与方向一致的 measured_move 投影；返回 (price, 说明)。"""
    wanted = ("range_up", "leg_up") if direction == "long" else ("range_down", "leg_down")
    for kind in wanted:
        for mm in moves:
            if getattr(mm, "kind", "") == kind:
                price = getattr(mm, "target_price", None)
                if price:
                    return float(price), f"{mm.kind} 投影（参考 {mm.reference}）"
    return None, ""


def plan_long(
    *,
    close: float,
    atr: float | None,
    features: Any,
    cfg: Any,
    tick: float | None = None,
) -> PricePlan:
    """构造做多价格计划。纯函数，不修改任何输入。"""
    return _plan("long", close=close, atr=atr, features=features, cfg=cfg, tick=tick)


def plan_short(
    *,
    close: float,
    atr: float | None,
    features: Any,
    cfg: Any,
    tick: float | None = None,
) -> PricePlan:
    """构造做空价格计划（与 plan_long 镜像）。"""
    return _plan("short", close=close, atr=atr, features=features, cfg=cfg, tick=tick)


def _plan(
    direction: str,
    *,
    close: float,
    atr: float | None,
    features: Any,
    cfg: Any,
    tick: float | None,
) -> PricePlan:
    short = direction == "short"
    sign = -1.0 if short else 1.0

    # 结构位：做多看支撑/失效位，做空看阻力/失效位
    if short:
        structure = features.resistances[0] if features.resistances else None
        invalidation = features.invalidation_short
    else:
        structure = features.supports[0] if features.supports else None
        invalidation = features.invalidation_long

    notes: list[str] = []
    entry_fb = stop_fb = target_fb = False

    if not structure and invalidation is None:
        return PricePlan(
            direction=direction,
            actionable=False,
            reason=(
                f"{'上方' if short else '下方'}无可用结构位"
                f"（支撑/阻力与失效位均缺失），无法给出可靠{'卖' if short else '买'}价"
            ),
            notes=notes,
        )

    # ── 挂单价
    if structure is not None and atr and atr > 0:
        entry = structure + sign * float(cfg.entry_offset_atr) * atr
        basis = f"{'阻力' if short else '支撑'} {structure} {'−' if short else '+'} {cfg.entry_offset_atr}×ATR"
    elif structure is not None:
        # 无 ATR：直接贴结构位
        entry = float(structure)
        basis = f"{'阻力' if short else '支撑'} {structure}（无 ATR，直接贴位）"
        notes.append("缺 ATR，挂单价未加缓冲偏移")
    elif atr and atr > 0:
        # 结构缺失 → ATR 兜底：从现价反向留出 1×ATR 的等待区
        entry = close - sign * float(cfg.fallback_stop_atr) * atr
        entry_fb = True
        basis = f"现价 {close} {'−' if short else '−'} {cfg.fallback_stop_atr}×ATR（结构位缺失，ATR 兜底）"
        notes.append("挂单价为 ATR 兜底值，非结构位")
    else:
        return PricePlan(
            direction=direction,
            actionable=False,
            reason="结构位与 ATR 均缺失，无法计算挂单价",
        )

    # ── 止损
    if invalidation is not None and atr and atr > 0:
        stop = invalidation - sign * float(cfg.stop_buffer_atr) * atr
        stop_basis = f"结构失效位 {invalidation} {'−' if not short else '+'} {cfg.stop_buffer_atr}×ATR"
    elif invalidation is not None:
        stop = float(invalidation)
        stop_basis = f"结构失效位 {invalidation}（无 ATR，未加缓冲）"
    elif atr and atr > 0:
        stop = entry + sign * float(cfg.fallback_stop_atr) * atr
        stop_fb = True
        stop_basis = f"挂单价 {'+' if short else '−'} {cfg.fallback_stop_atr}×ATR（失效位缺失，ATR 兜底）"
        notes.append("止损为 ATR 兜底值")
    else:
        stop = entry - sign * float(cfg.fallback_stop_atr) * max(abs(entry) * 0.01, 1e-9)
        stop_fb = True
        stop_basis = "按 1% 估算（无 ATR 无结构）"
        notes.append("止损不可靠：输入数据不足")

    # 防呆：止损必须真的在挂单的亏损侧
    if (direction == "long" and stop >= entry) or (direction == "short" and stop <= entry):
        return PricePlan(
            direction=direction,
            actionable=False,
            reason=(
                f"止损 {stop} 不在挂单价 {entry} 的亏损侧（结构位与现价关系异常），放弃出价"
            ),
        )

    # ── 目标
    mm_price, mm_desc = _pick_mm_target(getattr(features, "measured_moves", ()) or (), direction)
    if mm_price:
        target = float(mm_price)
        target_basis = mm_desc
    elif atr and atr > 0:
        r = abs(entry - stop)
        target = entry + sign * float(cfg.fallback_target_r) * r
        target_fb = True
        target_basis = f"entry {'+' if not short else '−'} {cfg.fallback_target_r}×R（无 MM 投影，R 倍数兜底）"
    else:
        target = None
        target_basis = "无 MM 且无 ATR，不给目标"

    risk = abs(entry - stop) if target is not None else None
    rr = None
    if risk and risk > 0 and target is not None:
        rr = abs(target - entry) / risk

    # 风险提示：结构失效位与挂单价贴得太近时，R 会小到没有统计意义
    if risk is not None and atr and atr > 0 and risk < 0.2 * atr:
        notes.append(
            f"止损距离仅 {risk / atr:.2f}×ATR（<0.2），R 偏小、盈亏比虚高，"
            "建议人工复核或放弃该方向"
        )
    # 风险提示：MM 投影目标动辄十几倍 ATR，盈亏比会虚高到没有操作意义
    if (
        target is not None
        and atr
        and atr > 0
        and abs(target - entry) > 10 * atr
    ):
        notes.append(
            f"目标距挂单价 {abs(target - entry) / atr:.1f}×ATR（>10，MM 长程投影），"
            "盈亏比虚高；建议分段止盈或以 2-3×ATR 的近端目标替代"
        )

    return PricePlan(
        direction=direction,
        actionable=True,
        entry=_round_price(entry, tick),
        stop=_round_price(stop, tick),
        target=_round_price(target, tick) if target is not None else None,
        risk_per_unit=_round_price(risk, tick) if risk is not None else None,
        rr_ratio=round(rr, 2) if rr is not None else None,
        entry_fallback=entry_fb,
        stop_fallback=stop_fb,
        target_fallback=target_fb,
        notes=tuple(notes) + (f"挂单依据：{basis}；止损依据：{stop_basis}；目标依据：{target_basis}",),
    )
