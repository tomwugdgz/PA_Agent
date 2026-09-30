# -*- coding: utf-8 -*-
"""贪婪调参器：用确定性回测当评估器，对定价参数做贪婪坐标下降。

算法（刻意选最简单、最可解释的）
--------------------------------
**贪婪坐标下降（Greedy Coordinate Descent）**

1. 以当前参数跑一次回测，作为基准目标值 ``total_r_net``（净收益 R 总和）；
2. 固定其他参数，逐个对可调参数尝试 ±delta（乘性步长，边界内截断）；
3. 候选若**严格优于**当前最优 → 立即接受（greedy：不回头、不探索），
   否则保持原值；
4. 一轮扫完所有参数后，步长收缩（0.5 → 0.25 → …），再扫一轮；
5. 全程每次评估都写入 journal 回测层（kind=tune_eval），总结写入 tuning 层。

为什么够用：回测是确定性纯函数（同一 frame + 同参数 = 同结果），
不存在随机噪声，贪婪法不会陷入"假改进"；代价是可能停在局部最优——
这正是用户要的"AI 自我调整优化"的诚实版本：**只接受回测证据支持的改进**。

边界（诚实声明）
----------------
* 评估器是**确定性规则回测**，不是 LLM；优化的是规则参数，
  不是 AI 的判断本身。
* 参数在**当前图表的历史窗口**上优化，可能过拟合该段行情——
  因此 GUI 只提供"应用为推荐值"，是否落盘由用户按键决定。
"""
from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pa_agent.journal.layer_journal import (
    JOURNAL_DIR, log_backtest, log_tuning,
)
from pa_agent.mt5trading.backtest import BacktestParams, run_backtest

logger = logging.getLogger(__name__)

#: 可调参数：(名字, 下界, 上界)。顺序即坐标下降的扫描顺序。
TUNABLE: tuple[tuple[str, float, float], ...] = (
    ("stop_buffer_atr", 0.05, 3.0),   # 止损缓冲：对胜率影响最大，先扫
    ("min_stop_atr", 0.2, 3.0),       # 最小止损距离
    ("target_r", 0.5, 10.0),          # 兜底目标 R 倍数
    ("entry_offset_atr", 0.0, 2.0),   # 挂单偏移
)

#: 每轮的乘性步长（候选 = 当前值 × (1 ± delta)），逐轮收缩
DEFAULT_DELTAS: tuple[float, ...] = (0.5, 0.25, 0.1)


@dataclass(frozen=True)
class TuningResult:
    """一轮贪婪调参的完整结果（GUI 展示 + journal 落盘共用）。"""

    symbol: str
    timeframe: str
    baseline_params: BacktestParams
    baseline_metrics: dict[str, Any]
    best_params: BacktestParams
    best_metrics: dict[str, Any]
    history: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    n_evals: int = 0

    @property
    def improved(self) -> bool:
        """最优是否严格优于基准（看净收益 R 总和）。"""
        return (self.best_metrics.get("total_r_net", 0.0)
                > self.baseline_metrics.get("total_r_net", 0.0) + 1e-9)

    def summary_text(self) -> str:
        """给 GUI 的中文总结。"""
        bm, mm = self.baseline_metrics, self.best_metrics
        lines = [
            f"调参对象：{self.symbol} {self.timeframe}"
            f"（评估 {self.n_evals} 次回测）",
            f"基准：总收益 {bm.get('total_r_net', 0):+.2f} R，"
            f"胜率 {bm.get('win_rate', 0) * 100:.1f}%，"
            f"盈亏比 {bm.get('profit_factor', 0):.2f}，"
            f"交易 {bm.get('n_trades', 0)} 笔",
        ]
        if self.improved:
            lines.append(
                f"最优：总收益 {mm.get('total_r_net', 0):+.2f} R，"
                f"胜率 {mm.get('win_rate', 0) * 100:.1f}%，"
                f"盈亏比 {mm.get('profit_factor', 0):.2f}，"
                f"交易 {mm.get('n_trades', 0)} 笔"
            )
            lines.append("接受的调整：")
            for h in self.history:
                if h.get("accepted"):
                    lines.append(
                        f"  · {h['param']}: {h['old']:g} → {h['candidate']:g}"
                        f"（总收益 {h['objective']:+.2f} R）"
                    )
        else:
            lines.append("结论：当前参数已是本轮扫描的最优，无需调整。")
        lines.append(
            "⚠️ 结果基于当前图表历史窗口，存在过拟合风险；应用前请在多段行情复验。"
        )
        return "\n".join(lines)

    def changed_params(self) -> dict[str, tuple[float, float]]:
        """被接受的参数变化 {名字: (旧值, 新值)}。"""
        out: dict[str, tuple[float, float]] = {}
        for h in self.history:
            if h.get("accepted"):
                out[h["param"]] = (h["old"], h["candidate"])
        return out


def _objective(metrics: dict[str, Any]) -> float:
    """调参目标函数：净收益 R 总和（正比真实盈亏，其他指标只做展示）。"""
    return float(metrics.get("total_r_net", 0.0))


def _candidate(
    params: BacktestParams, name: str, delta: float, sign: int
) -> BacktestParams | None:
    """生成某参数在 ±delta 乘性步长下的候选（越界截断，无变化返回 None）。"""
    lo, hi = next((l, h) for n, l, h in TUNABLE if n == name)
    cur = float(getattr(params, name))
    val = cur * (1.0 + sign * delta)
    val = round(min(hi, max(lo, val)), 4)
    if abs(val - cur) < 1e-9:
        return None
    return dataclasses.replace(params, **{name: val})


def greedy_tune(
    frame: Any,
    base_params: BacktestParams,
    *,
    deltas: tuple[float, ...] = DEFAULT_DELTAS,
    journal: bool = True,
) -> TuningResult:
    """在 frame 上做贪婪坐标下降调参。纯计算 + journal 落盘，不碰 MT5。

    Raises:
        ValueError: K 线数量不足（由 run_backtest 抛出，原样上抛给 GUI 展示）。
    """
    baseline = run_backtest(frame, base_params)
    best_params, best_metrics = base_params, baseline.metrics
    best_obj = _objective(best_metrics)
    history: list[dict[str, Any]] = []
    n_evals = 1

    if journal:
        log_backtest(
            symbol=frame.symbol, timeframe=frame.timeframe, kind="backtest_run",
            params=dataclasses.asdict(base_params), metrics=baseline.metrics,
            extra={"role": "tune_baseline"},
        )

    for round_no, delta in enumerate(deltas, start=1):
        for name, _lo, _hi in TUNABLE:
            for sign in (+1, -1):
                cand = _candidate(best_params, name, delta, sign)
                if cand is None:
                    continue
                try:
                    result = run_backtest(frame, cand)
                except ValueError:
                    continue  # 候选参数导致窗口不足等，跳过
                n_evals += 1
                obj = _objective(result.metrics)
                accepted = obj > best_obj + 1e-9
                history.append({
                    "round": round_no, "delta": delta, "param": name,
                    "direction": "up" if sign > 0 else "down",
                    "old": float(getattr(best_params, name)),
                    "candidate": float(getattr(cand, name)),
                    "objective": round(obj, 4),
                    "accepted": accepted,
                })
                if journal:
                    log_backtest(
                        symbol=frame.symbol, timeframe=frame.timeframe,
                        kind="tune_eval",
                        params=dataclasses.asdict(cand),
                        metrics=result.metrics,
                        extra={"accepted": accepted, "delta": delta},
                    )
                if accepted:
                    best_params, best_metrics, best_obj = (
                        cand, result.metrics, obj,
                    )
                    break  # greedy：该参数接受后立即换下一个参数

    out = TuningResult(
        symbol=str(frame.symbol),
        timeframe=str(frame.timeframe),
        baseline_params=base_params,
        baseline_metrics=baseline.metrics,
        best_params=best_params,
        best_metrics=best_metrics,
        history=tuple(history),
        n_evals=n_evals,
    )

    if journal:
        log_tuning(
            symbol=out.symbol, timeframe=out.timeframe,
            baseline=dataclasses.asdict(base_params),
            best=dataclasses.asdict(best_params),
            accepted=[h for h in history if h.get("accepted")],
            n_evals=n_evals,
        )
        _write_tuned_params(out)
    return out


def _write_tuned_params(result: TuningResult) -> Path | None:
    """把本轮最优参数持久化到 tuned_params.json（下次人工或自动对照）。"""
    try:
        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
        path = JOURNAL_DIR / "tuned_params.json"
        payload = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ts_ms": int(time.time() * 1000),
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "improved": result.improved,
            "baseline": dataclasses.asdict(result.baseline_params),
            "best": dataclasses.asdict(result.best_params),
            "changed": {k: {"old": o, "new": n}
                        for k, (o, n) in result.changed_params().items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path
    except Exception as exc:  # noqa: BLE001
        logger.debug("tuned_params.json 写入失败: %s", exc)
        return None


def apply_to_settings(settings: Any, result: TuningResult) -> list[str]:
    """把最优参数写回 settings.laya 并返回变更说明（不含持久化）。

    映射关系：entry_offset_atr / stop_buffer_atr 同名直写；
    回测的 target_r 对应 settings.laya.fallback_target_r（兜底目标）；
    min_stop_atr 是回测层专用参数，settings.laya 无对应字段，不写。
    """
    laya = getattr(settings, "laya", None)
    if laya is None:
        return ["settings.laya 不存在，未应用"]
    changed: list[str] = []
    mapping = {
        "entry_offset_atr": "entry_offset_atr",
        "stop_buffer_atr": "stop_buffer_atr",
        "target_r": "fallback_target_r",
    }
    for bt_name, setting_name in mapping.items():
        new_val = float(getattr(result.best_params, bt_name))
        old_val = float(getattr(laya, setting_name))
        if abs(new_val - old_val) > 1e-9:
            setattr(laya, setting_name, new_val)
            changed.append(f"laya.{setting_name}: {old_val:g} → {new_val:g}")
    if not changed:
        changed.append("参数无变化，未修改设置")
    return changed
