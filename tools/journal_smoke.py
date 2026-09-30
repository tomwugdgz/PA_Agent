# -*- coding: utf-8 -*-
"""journal 五层记录 + 贪婪调参 冒烟测试（无 GUI、无网络）。

跑法：python tools/journal_smoke.py
验证点：
1. 五层便捷入口都能落盘（data/decision/risk/backtest/execution/tuning）
2. 贪婪调参在合成数据上完整跑完并产出 tuned_params.json
3. stats_text / tail 读侧正常
"""
from __future__ import annotations

import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pa_agent.data.base import KlineBar, KlineFrame, IndicatorBundle
from pa_agent.journal import layer_journal as LJ
from pa_agent.journal.greedy_tuner import TuningResult, greedy_tune
from pa_agent.mt5trading.backtest import BacktestParams, run_backtest


def make_frame(n: int = 400) -> KlineFrame:
    """合成有结构的数据：分段趋势 + 噪声（与回测冒烟同思路）。"""
    rng = random.Random(42)
    bars: list[KlineBar] = []
    price = 100.0
    drift = 0.0
    bars_chrono: list[KlineBar] = []
    for i in range(n):
        if i % 40 == 0:
            drift = rng.choice((-0.35, 0.0, 0.4))   # 分段趋势
        o = price
        moves = [o + rng.gauss(drift, 0.5) for _ in range(4)]
        c = moves[-1]
        hi = max(o, *moves) + abs(rng.gauss(0, 0.2))
        lo = min(o, *moves) - abs(rng.gauss(0, 0.2))
        ts = (1700000000 + i * 3600) * 1000
        bars_chrono.append(KlineBar(seq=n - i, ts_open=ts, open=round(o, 2),
                                    high=round(hi, 2), low=round(lo, 2),
                                    close=round(c, 2), volume=1000.0,
                                    closed=True))
        price = c

    atrs = []
    for i in range(len(bars_chrono)):
        seg = bars_chrono[max(0, i - 13):i + 1]
        trs = [max(b.high - b.low, abs(b.high - p.close), abs(b.low - p.close))
               for b, p in zip(seg[1:], seg[:-1])] or [1.0]
        atrs.append(round(sum(trs) / len(trs), 4))

    newest_first = list(reversed(bars_chrono))
    # seq 约定：最新一根 seq=1，往旧递增
    newest_first = [
        KlineBar(seq=i + 1, ts_open=b.ts_open, open=b.open, high=b.high,
                 low=b.low, close=b.close, volume=b.volume, closed=b.closed)
        for i, b in enumerate(newest_first)
    ]
    atr_newest_first = tuple(reversed(atrs))
    return KlineFrame(
        symbol="TEST/SMOKE", timeframe="H1",
        bars=newest_first,
        indicators=IndicatorBundle(ema20=tuple(0.0 for _ in newest_first),
                                   atr14=atr_newest_first),
        snapshot_ts_local_ms=0,
    )


def main() -> None:
    frame = make_frame()
    sym, tf = frame.symbol, frame.timeframe

    # ── 1) 五层便捷入口 ──────────────────────────────────────────────
    ok1 = LJ.log_data(symbol=sym, timeframe=tf, n_bars=400, close=100.5,
                      atr=1.2, device="cpu", latency_ms=12.3, n_errors=0)
    ok2 = LJ.log_decision(symbol=sym, timeframe=tf,
                          answers={"方向": {"kind": "choice", "value": "long",
                                            "confidence": 0.42}},
                          plans={"long": {"actionable": True, "entry": 100.1,
                                          "stop": 98.8, "target": 102.7,
                                          "rr": 2.0}})
    ok3 = LJ.log_risk(symbol=sym, timeframe=tf,
                      warnings=["止损距离仅 0.5×ATR（<0.2），R 偏小"],
                      low_confidence=True, max_rr=2.0)
    ok4 = LJ.log_execution(symbol=sym, ok=True, retcode=10009,
                           direction="long", order_kind="limit",
                           entry=100.1, stop_loss=98.8, lot=0.01,
                           ticket=123456, message="限价单成交")
    print(f"五层写入: data={ok1} decision={ok2} risk={ok3} execution={ok4}")
    assert ok1 and ok2 and ok3 and ok4

    # ── 2) 贪婪调参（2 轮步长，控制时长） ────────────────────────────
    base = BacktestParams(lookback=100, cost_points=10)
    base_result = run_backtest(frame, base)
    print(f"基准回测: {base_result.metrics['n_trades']} 笔, "
          f"总收益 {base_result.metrics['total_r_net']:+.2f} R")
    res = greedy_tune(frame, base, deltas=(0.5, 0.25))
    assert isinstance(res, TuningResult)
    print(res.summary_text())
    tuned = Path(LJ.JOURNAL_DIR) / "tuned_params.json"
    print(f"tuned_params.json 存在: {tuned.is_file()}")

    # ── 3) 读侧 ──────────────────────────────────────────────────────
    stats = LJ.stats_text(days=7)
    print("--- stats_text ---")
    print(stats)
    for layer in ("data", "decision", "risk", "backtest", "execution", "tuning"):
        rows = LJ.tail(layer, n=3)
        assert rows, f"{layer} 层 tail 为空"
    print("tail 六层全部可读")
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
