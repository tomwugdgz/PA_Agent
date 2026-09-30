# -*- coding: utf-8 -*-
"""确定性回测引擎：与 Laya 报告/AI 决策**同一套定价口径**在历史 K 线上走查。

重要边界
--------
本回测回放的是**确定性规则**（结构位 + ATR），不是 DeepSeek 大模型——
LLM 无法在回测里逐根调用（成本与延迟都不可行）。因此结果回答的问题是：
「AI 决策所依赖的那套结构性规则，历史上表现如何」，而不是「AI 本身的胜率」。

信号集（与 AI 订单类型一一对应，全部可开关）
--------------------------------------------
* 限价单（回踩反弹做多 / 反抽阻力做空）：近端测试过支撑且未破位、非铁丝网
* 突破单（收盘突破确认后入场）：breakout_quality ∈ {close_breakout, surviving}
* 止损 = 结构失效位 ∓ stop_buffer×ATR；目标 = MM 投影，否则 entry + target_r×R
* 同根 K 线同时触发 TP 与 SL → 保守记亏损；超时按收盘价离场
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pa_agent.ai.market_features import compute_simple_market_features
from pa_agent.data.base import IndicatorBundle, KlineFrame


@dataclass(frozen=True)
class BacktestParams:
    """回测参数（GUI 面板可调，默认与 settings.mt5trading 对齐）。"""

    lookback: int = 100
    entry_offset_atr: float = 0.10
    stop_buffer_atr: float = 0.25
    #: 最小止损距离（×ATR）：结构失效位常与挂单结构位同点，缓冲必须盖过噪音，
    #: 否则 R 小于波动幅度、止损必然先于目标触发（合成数据实测教训）
    min_stop_atr: float = 1.0
    target_r: float = 2.0
    #: 近端测试窗口：最近 N 根内触碰过结构位才算「回踩」
    touch_window: int = 3
    pending_expiry_bars: int = 12
    timeout_bars: int = 50
    #: 单边成本（点），乘 2 计往返
    cost_points: int = 10
    #: 信号开关
    enable_limit: bool = True       # 限价单：回踩
    enable_breakout: bool = True    # 突破单
    #: 回测起始：跳过前 warmup 根（窗口积累）
    warmup_bars: int = 60


@dataclass(frozen=True)
class Trade:
    """一笔已平仓交易。R = 盈亏 / 初始风险。"""

    side: str                 # long | short
    kind: str                 # limit | breakout | timeout_exit
    entry_ts_ms: int
    exit_ts_ms: int
    entry: float
    stop: float
    target: float
    exit_price: float
    bars_held: int
    outcome: str              # tp | sl | timeout | ambiguous
    r_gross: float
    r_net: float              # 扣除成本后


@dataclass(frozen=True)
class BacktestResult:
    symbol: str
    timeframe: str
    bars_tested: int
    params: BacktestParams
    trades: tuple[Trade, ...]
    metrics: dict[str, Any]
    equity_curve: tuple[tuple[int, float], ...]   # (exit_ts_ms, 累计 R)

    def summary_text(self) -> str:
        """给 GUI / 报告用的一段中文摘要。"""
        m = self.metrics
        lines = [
            f"回测区间：{self.bars_tested} 根（{self.symbol} {self.timeframe}）",
            f"交易数：{m['n_trades']}（多头 {m['n_long']} / 空头 {m['n_short']}）",
            f"胜率：{m['win_rate'] * 100:.1f}%　总收益：{m['total_r_net']:+.2f} R",
            f"盈利因子：{m['profit_factor']:.2f}　平均 R：{m['avg_r_net']:+.3f}",
            f"最大回撤：{m['max_drawdown_r']:.2f} R",
            f"出场分布：TP {m['n_tp']} / SL {m['n_sl']} / 超时 {m['n_timeout']}",
            f"参数：回看 {self.params.lookback} 根，止损缓冲 {self.params.stop_buffer_atr}×ATR，"
            f"目标 {self.params.target_r}×R，成本 {self.params.cost_points} 点/边",
        ]
        return "\n".join(lines)


# ── 主入口 ───────────────────────────────────────────────────────────────────


def run_backtest(frame: KlineFrame, params: BacktestParams) -> BacktestResult:
    """在 frame 上跑确定性规则回测。纯计算，不碰 MT5、不碰网络。

    frame.bars 为最新在前（PA_Agent 全局约定）；内部转成时间升序处理。
    """
    chrono = list(reversed(frame.bars))          # 旧 → 新
    n = len(chrono)
    if n < params.warmup_bars + params.lookback:
        raise ValueError(
            f"K 线不足：{n} 根，至少需要 {params.warmup_bars + params.lookback} 根"
            f"（warmup {params.warmup_bars} + 窗口 {params.lookback}）"
        )

    # 全帧 ATR 一次算好，窗口内复用（与 indicators.atr14 同源，避免每窗口重算）
    atr_by_chrono = _atr_chrono(frame, n)

    trades: list[Trade] = []
    equity: list[tuple[int, float]] = []
    cum_r = 0.0

    point = _infer_point(chrono)
    cost_price = 2 * params.cost_points * point      # 往返成本（价格单位）

    pending: dict[str, Any] | None = None            # 挂单
    position: dict[str, Any] | None = None           # 持仓
    cooldown_until = -1

    for i in range(params.warmup_bars, n):
        bar = chrono[i]
        hi, lo, close = float(bar.high), float(bar.low), float(bar.close)

        # ── 1) 持仓管理（先于新信号：同一根先处理离场）
        if position is not None:
            res = _check_exit(position, hi, lo, close, i, params)
            if res is not None:
                exit_price, outcome, bars_held = res
                t = _close_trade(position,
                                 chrono[position["entry_bar"]].ts_open,   # 入场根时间
                                 bar.ts_open, exit_price, outcome, bars_held,
                                 cost_price, atr_by_chrono)
                trades.append(t)
                cum_r += t.r_net
                equity.append((t.exit_ts_ms, round(cum_r, 4)))
                position = None
                cooldown_until = i + 1
            # 持仓期间不看新信号（单持仓模型）

        # ── 2) 挂单管理
        if pending is not None and position is None:
            fill = _check_fill(pending, hi, lo)
            if fill is not None:
                position = dict(pending, entry=fill, entry_ts_ms=bar.ts_open,
                                entry_bar=i, kind=pending["kind"])
                pending = None
            elif i - pending["placed_bar"] >= params.pending_expiry_bars:
                pending = None   # 到期撤单

        # ── 3) 新信号（无持仓、无挂单、非冷却期）
        if position is None and pending is None and i > cooldown_until:
            sig = _signal_at(frame, chrono, i, atr_by_chrono[i], params)
            if sig is not None:
                pending = dict(sig, placed_bar=i)

    # 期末强平（按最后一根收盘）
    if position is not None:
        last = chrono[-1]
        bars_held = len(chrono) - 1 - position["entry_bar"]
        t = _close_trade(position, chrono[position["entry_bar"]].ts_open, last.ts_open,
                         float(last.close), "timeout", bars_held, cost_price, atr_by_chrono)
        trades.append(t)
        cum_r += t.r_net
        equity.append((t.exit_ts_ms, round(cum_r, 4)))

    return BacktestResult(
        symbol=frame.symbol,
        timeframe=frame.timeframe,
        bars_tested=n - params.warmup_bars,
        params=params,
        trades=tuple(trades),
        metrics=_metrics(trades),
        equity_curve=tuple(equity),
    )


# ── ATR ──────────────────────────────────────────────────────────────────────


def _atr_chrono(frame: KlineFrame, n: int) -> list[float | None]:
    """把 frame.indicators.atr14（最新在前）映射成时间升序列表。"""
    atr14 = frame.indicators.atr14 if frame.indicators else ()
    out: list[float | None] = []
    for i in range(n):                      # i=0 是最旧
        v = atr14[n - 1 - i] if n - 1 - i < len(atr14) else None
        out.append(float(v) if v is not None and v == v and v > 0 else None)
    return out


def _infer_point(chrono: list[Any]) -> float:
    """从最近 30 根的小数位数推断 point（与 price_tick 同思路，避免循环依赖）。"""
    max_dec = 0
    for bar in chrono[-30:]:
        for v in (bar.open, bar.high, bar.low, bar.close):
            text = f"{float(v):.12f}".rstrip("0")
            if "." in text:
                max_dec = max(max_dec, len(text.split(".")[1]))
    return 10 ** (-min(max_dec, 6)) if max_dec else 1.0


# ── 信号 ─────────────────────────────────────────────────────────────────────


def _signal_at(
    frame: KlineFrame,
    chrono: list[Any],
    i: int,
    atr: float | None,
    params: BacktestParams,
) -> dict[str, Any] | None:
    """在 chrono[i]（刚收盘）评估多空信号。返回挂单描述或 None。"""
    if atr is None or atr <= 0:
        return None
    lo_idx = max(0, i - params.lookback + 1)
    window = chrono[lo_idx:i + 1]                     # 旧 → 新
    win_frame = KlineFrame(
        symbol=frame.symbol,
        timeframe=frame.timeframe,
        bars=tuple(reversed(window)),                 # 特征函数要最新在前
        indicators=IndicatorBundle(
            ema20=tuple(0.0 for _ in window),
            atr14=tuple(atr for _ in window),
        ),
        snapshot_ts_local_ms=0,
    )
    f = compute_simple_market_features(win_frame)
    close = float(window[-1].close)

    if f.barbwire_candidate:
        return None                                    # 铁丝网：不做

    recent_lows = [float(b.low) for b in window[-params.touch_window:]]
    recent_highs = [float(b.high) for b in window[-params.touch_window:]]

    # ── 多头：限价（回踩支撑）优先，其次突破
    support = f.supports[0] if f.supports else None
    resistance = f.resistances[0] if f.resistances else None
    mm_up = _pick_mm(f.measured_moves, up=True)
    mm_down = _pick_mm(f.measured_moves, up=False)

    if support is not None and params.enable_limit:
        tested = min(recent_lows) <= support + 0.5 * atr
        holding = close > support                      # 收盘仍在支撑上方 = 未破位
        sideways = f.breakout_quality in ("none", "testing", "failed")
        if tested and holding and sideways:
            entry = support + params.entry_offset_atr * atr
            stop = support - max(params.stop_buffer_atr, params.min_stop_atr) * atr
            target = mm_up if mm_up else entry + params.target_r * (entry - stop)
            if stop < entry < target:
                return dict(side="long", kind="limit", entry=entry, stop=stop,
                            target=target, sig_bar=i)
    if resistance is not None and params.enable_breakout:
        if f.breakout_quality in ("close_breakout", "surviving") and close > resistance:
            entry = close                              # 突破确认：收盘市价入场
            stop = min(resistance - max(params.stop_buffer_atr, params.min_stop_atr) * atr, entry - 0.5 * atr)
            target = mm_up if mm_up else entry + params.target_r * (entry - stop)
            if stop < entry < target:
                return dict(side="long", kind="breakout", entry=entry, stop=stop,
                            target=target, sig_bar=i)

    # ── 空头镜像
    if resistance is not None and params.enable_limit:
        tested = max(recent_highs) >= resistance - 0.5 * atr
        holding = close < resistance
        sideways = f.breakout_quality in ("none", "testing", "failed")
        if tested and holding and sideways:
            entry = resistance - params.entry_offset_atr * atr
            stop = resistance + max(params.stop_buffer_atr, params.min_stop_atr) * atr
            target = mm_down if mm_down else entry - params.target_r * (stop - entry)
            if target < entry < stop:
                return dict(side="short", kind="limit", entry=entry, stop=stop,
                            target=target, sig_bar=i)
    if support is not None and params.enable_breakout:
        if f.breakout_quality in ("close_breakout", "surviving") and close < support:
            entry = close
            stop = max(support + max(params.stop_buffer_atr, params.min_stop_atr) * atr, entry + 0.5 * atr)
            target = mm_down if mm_down else entry - params.target_r * (stop - entry)
            if target < entry < stop:
                return dict(side="short", kind="breakout", entry=entry, stop=stop,
                            target=target, sig_bar=i)
    return None


def _pick_mm(moves: tuple[Any, ...], *, up: bool) -> float | None:
    wanted = ("range_up", "leg_up") if up else ("range_down", "leg_down")
    for kind in wanted:
        for mm in moves:
            if getattr(mm, "kind", "") == kind:
                v = getattr(mm, "target_price", None)
                if v:
                    return float(v)
    return None


# ── 撮合与出场 ───────────────────────────────────────────────────────────────


def _check_fill(p: dict[str, Any], hi: float, lo: float) -> float | None:
    """挂单是否在本根成交。限价=价格回摸；突破=价格越过入场。"""
    if p["side"] == "long":
        if p["kind"] == "limit":
            return float(p["entry"]) if lo <= p["entry"] else None
        return float(p["entry"]) if hi >= p["entry"] else None
    if p["kind"] == "limit":
        return float(p["entry"]) if hi >= p["entry"] else None
    return float(p["entry"]) if lo <= p["entry"] else None


def _check_exit(
    pos: dict[str, Any], hi: float, lo: float, close: float,
    i: int, params: BacktestParams,
) -> tuple[float, str, int] | None:
    """持仓出场判定。返回 (exit_price, outcome, bars_held) 或 None。

    注意：入场当根不做出场判定（信号根与成交根可能同根，这里成交后从下一根开始管）。
    """
    bars_held = i - pos["entry_bar"]
    if bars_held <= 0:
        return None
    if pos["side"] == "long":
        hit_tp = hi >= pos["target"]
        hit_sl = lo <= pos["stop"]
    else:
        hit_tp = lo <= pos["target"]
        hit_sl = hi >= pos["stop"]

    if hit_tp and hit_sl:
        return (float(pos["stop"]), "ambiguous", bars_held)     # 保守记亏
    if hit_tp:
        return (float(pos["target"]), "tp", bars_held)
    if hit_sl:
        return (float(pos["stop"]), "sl", bars_held)
    if bars_held >= params.timeout_bars:
        return (close, "timeout", bars_held)
    return None


def _close_trade(
    pos: dict[str, Any], entry_ts: int, exit_ts: int,
    exit_price: float, outcome: str, bars_held: int,
    cost_price: float, atr_by_chrono: list[float | None],
) -> Trade:
    entry, stop = float(pos["entry"]), float(pos["stop"])
    risk = abs(entry - stop)
    if pos["side"] == "long":
        pnl = exit_price - entry
    else:
        pnl = entry - exit_price
    r_gross = pnl / risk if risk > 0 else 0.0
    r_net = (pnl - cost_price) / risk if risk > 0 else 0.0
    return Trade(
        side=pos["side"], kind=pos["kind"],
        entry_ts_ms=int(entry_ts), exit_ts_ms=int(exit_ts),
        entry=round(entry, 8), stop=round(stop, 8),
        target=round(float(pos["target"]), 8), exit_price=round(exit_price, 8),
        bars_held=bars_held, outcome=outcome,
        r_gross=round(r_gross, 4), r_net=round(r_net, 4),
    )


# ── 统计 ─────────────────────────────────────────────────────────────────────


def _metrics(trades: list[Trade]) -> dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "n_long": 0, "n_short": 0, "win_rate": 0.0,
                "total_r_net": 0.0, "avg_r_net": 0.0, "profit_factor": 0.0,
                "max_drawdown_r": 0.0, "n_tp": 0, "n_sl": 0, "n_timeout": 0}
    wins = [t for t in trades if t.r_net > 0]
    losses = [t for t in trades if t.r_net <= 0]
    gross_win = sum(t.r_net for t in wins)
    gross_loss = abs(sum(t.r_net for t in losses))
    cum, peak, mdd = 0.0, 0.0, 0.0
    for t in trades:
        cum += t.r_net
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    return {
        "n_trades": n,
        "n_long": sum(1 for t in trades if t.side == "long"),
        "n_short": sum(1 for t in trades if t.side == "short"),
        "win_rate": len(wins) / n,
        "total_r_net": round(cum, 4),
        "avg_r_net": round(cum / n, 4),
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else float("inf"),
        "max_drawdown_r": round(mdd, 4),
        "n_tp": sum(1 for t in trades if t.outcome == "tp"),
        "n_sl": sum(1 for t in trades if t.outcome in ("sl", "ambiguous")),
        "n_timeout": sum(1 for t in trades if t.outcome == "timeout"),
    }
