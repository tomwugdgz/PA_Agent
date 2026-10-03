# -*- coding: utf-8 -*-
"""弱监督自动标注 CLI：用未来真实走势给历史 Laya 样本打标签。

用法
----
    # 看有多少待标注
    python tools/weak_label.py --status

    # 回填所有可判定的历史样本（自动按样本涉及的品种/周期批量拉取）
    python tools/weak_label.py

    # 只看会标多少，不写文件
    python tools/weak_label.py --dry-run

    # 清掉所有弱监督标签（人工标签保留），用于标错后重来
    python tools/weak_label.py --reset-weak

流程
----
1. 扫描全部未标注样本，按 ``(symbol, timeframe)`` 汇总时间跨度
2. 用 ``mt5.copy_rates_range`` **按时间范围**回溯拉 K 线
   （``latest_snapshot`` 只能拿「现在往前 N 根」，1m 品种 3000 根仅覆盖 2 天）
3. 对每条样本，找它决策时刻之后的走势
4. 用 Kaufman 效率比 + ATR 阈值判定正确答案
5. 原地回填 ``label``，标记 ``source="weak_supervision"``

已知阻断项
----------
经纪商更换报价源/合约规格后，历史 close 与当前行情不再同源
（实测 XAUUSD 样本 2001.06 vs 现价 4259.91，缺口达数百 ATR）。
这类样本由 ``max_gap_atr`` 缺口守卫拦下并计入 ``gap_too_large``，
不会写入脏标签。可行路径是**人工标注**或**从现在起持续积累新样本**。

样本足够后接着跑：``python tools/calibrate_laya.py``
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pa_agent.ai.laya_annotation import annotations_dir, stats  # noqa: E402
from pa_agent.ai.laya_weaklabel import (  # noqa: E402
    DEFAULT_ATR_MULT,
    label_summary,
    needed_series,
    relabel_unlabeled,
)


#: 项目内部 timeframe 名 → MT5 常量后缀
_TF_SUFFIX = {
    "M1": "TIMEFRAME_M1", "M3": "TIMEFRAME_M3", "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15", "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1", "W1": "TIMEFRAME_W1",
    "1m": "TIMEFRAME_M1", "3m": "TIMEFRAME_M3", "5m": "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15", "30m": "TIMEFRAME_M30",
    "1h": "TIMEFRAME_H1", "4h": "TIMEFRAME_H4",
    "1d": "TIMEFRAME_D1", "1w": "TIMEFRAME_W1",
}
_TF_MS = {
    "M1": 60_000, "M5": 300_000, "M15": 900_000, "M30": 1_800_000,
    "H1": 3_600_000, "H4": 14_400_000, "D1": 86_400_000, "W1": 604_800_000,
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000, "1w": 604_800_000,
}


def _load_bridge_by_path():
    """按文件路径加载 ``mt5_bridge`` 并返回其 ``connect``；失败返回 None。

    不能用 ``importlib.util.find_spec``——它同样会执行父包
    ``pa_agent/mt5trading/__init__.py``，照样被 PyQt6 拖挂。
    另外必须先把模块注册进 ``sys.modules``：文件里的 ``@dataclass``
    会回查 ``sys.modules[cls.__module__].__dict__``，未注册会抛
    ``'NoneType' object has no attribute '__dict__'``。
    """
    import importlib.util

    bridge_path = Path(__file__).resolve().parent.parent / "pa_agent" / "mt5trading" / "mt5_bridge.py"
    if not bridge_path.is_file():
        print(f"[错误] 找不到 {bridge_path}", file=sys.stderr)
        return None
    spec = importlib.util.spec_from_file_location("_laya_weak_mt5_bridge", bridge_path)
    if spec is None or spec.loader is None:
        print("[错误] 无法为 mt5_bridge 构造 spec", file=sys.stderr)
        return None
    bridge = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bridge
    try:
        spec.loader.exec_module(bridge)
    except Exception as exc:  # noqa: BLE001
        sys.modules.pop(spec.name, None)
        print(f"[错误] 加载 mt5_bridge 失败：{exc}", file=sys.stderr)
        return None
    return getattr(bridge, "connect", None)


def _tf_of(timeframe: str) -> tuple[int, int]:
    """返回 (MT5 timeframe 常量, 单根毫秒)。未知周期抛 KeyError。"""
    suffix = _TF_SUFFIX[str(timeframe)]
    import MetaTrader5 as mt5

    return int(getattr(mt5, suffix)), _TF_MS[str(timeframe)]


def _fetch_span(symbol: str, timeframe: str, since_ms: int, until_ms: int) -> list[Any]:
    """按**时间范围**拉 K 线（而非只取最新 N 根）。

    这是覆盖率的关键：``MT5Source.latest_snapshot(n)`` 只能拿到「现在往前 N 根」，
    对 1m 品种而言 3000 根只覆盖 2 天，几周前的样本完全取不到（实测退化到
    close 匹配，匹配到错误位置，close 与未来第一根差 11 倍 ATR）。
    弱监督必须能回溯到样本当时的行情，所以这里直接用
    ``copy_rates_range(symbol, tf, from, to)``。
    """
    try:
        import MetaTrader5 as mt5
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] 导入 MT5 失败：{exc}", file=sys.stderr)
        return []

    # 优先正常导入；失败再退回「按文件路径加载 mt5_bridge」。
    # 退化路径的必要性：包级 `pa_agent.mt5trading.__init__` 会经
    # backtest → market_features → util.event_bus → PyQt6，
    # 在没装 PyQt6 的解释器里直接 ImportError。弱监督只需要连接能力。
    connect = None
    try:
        from pa_agent.mt5trading.mt5_bridge import connect as _connect

        connect = _connect
    except Exception as exc:  # noqa: BLE001
        print(f"[提示] 包级导入失败（{exc}），改用按文件加载 mt5_bridge")
        connect = _load_bridge_by_path()
        if connect is None:
            return []

    try:
        connect("")
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] 连接 MT5 失败：{exc}", file=sys.stderr)
        return []
    try:
        tf, bar_ms = _tf_of(timeframe)
    except (KeyError, AttributeError) as exc:
        print(f"[警告] {symbol} 周期 {timeframe} 无法识别：{exc}")
        return []

    # 多取一段前置窗口，确保最早样本也有「决策前」的上下文
    pad = 60 * bar_ms
    try:
        mt5.symbol_select(symbol, True)
    except Exception:  # noqa: BLE001
        pass
    rates = mt5.copy_rates_range(symbol, tf,
                                 int((since_ms - pad) // 1000), int(until_ms // 1000))
    if rates is None or len(rates) == 0:
        return []

    from pa_agent.data.base import KlineBar

    n = len(rates)
    bars: list[Any] = []
    for i, r in enumerate(rates):
        try:
            bars.append(KlineBar(
                seq=n - i,                # 时间正序→ 序号倒序（seq=1 最新）
                ts_open=int(r["time"]) * 1000,
                open=float(r["open"]), high=float(r["high"]),
                low=float(r["low"]), close=float(r["close"]),
                volume=float(r["tick_volume"]),
            ))
        except (TypeError, ValueError, KeyError):
            continue
    if not bars:
        return []
    holder = _SymbolTimeframe(symbol, timeframe)
    holder.bars = bars
    return [holder]


def _sample_time_range(experience_dir: Path | None = None
                       ) -> dict[tuple[str, str], tuple[int, int]]:
    """统计每个 ``(symbol, timeframe)`` 下未标注样本的最早/最晚决策时刻。"""
    from pa_agent.ai.laya_annotation import _iter_samples

    rng: dict[tuple[str, str], list[int]] = {}
    for s in _iter_samples(annotations_dir(experience_dir)):
        if s.get("label"):
            continue
        ctx = s.get("context") or {}
        sym, tf, ts = ctx.get("symbol"), ctx.get("timeframe"), s.get("ts_ms")
        if not (sym and tf and ts is not None):
            continue
        key = (str(sym), str(tf))
        t = int(ts)
        cur = rng.get(key)
        if cur is None:
            rng[key] = [t, t]
        else:
            cur[0] = min(cur[0], t)
            cur[1] = max(cur[1], t)
    return {k: (v[0], v[1]) for k, v in rng.items()}


class _SymbolTimeframe:
    """轻量frame 容器：只带索引用到的三个字段。"""

    def __init__(self, symbol: str, timeframe: str) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.bars: list[Any] = []


def _fmt_ts(ms: int) -> str:
    """毫秒时间戳→ ``MM-DD HH:mm``（本机时区）。"""
    return time.strftime("%m-%d %H:%M", time.localtime(ms / 1000))


def _clear_weak_labels() -> int:
    """清除所有 ``weak_supervision`` 来源的标签（人工标签保留）。

    改了判定规则后必须先清干净再重跑，否则新旧规则产生的标签会混在一起，
    校准时等于喂了自相矛盾的监督信号。返回清除条数。
    """
    out_dir = annotations_dir()
    n = 0
    for path in sorted(out_dir.glob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        changed = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(s.get("source")) == "weak_supervision":
                s["label"] = None
                s["source"] = "auto"
                s.pop("labeled_at_ms", None)
                s.pop("label_method", None)
                lines[i] = json.dumps(s, ensure_ascii=False)
                changed += 1
        if changed:
            try:
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                n += changed
            except OSError as exc:
                print(f"[警告] 清除失败 {path}: {exc}", file=sys.stderr)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Laya 弱监督自动标注")
    ap.add_argument("--symbol", default="",
                    help="只处理指定品种（默认自动处理全部待标注品种）")
    ap.add_argument("--timeframe-override", default="",
                    help="把所有品种的周期强制改成这个（调试用，例：15m）")
    ap.add_argument("--atr-mult", type=float, default=DEFAULT_ATR_MULT,
                    help="净涨跌超过 N×ATR 才算有方向")
    ap.add_argument("--reset-weak", action="store_true",
                    help="先清除已有的 weak_supervision 标签再重跑"
                         "（改了判定规则后用，避免旧脏标签残留）")
    ap.add_argument("--dry-run", action="store_true", help="不写文件，只统计")
    ap.add_argument("--status", action="store_true", help="只看样本状态，不连 MT5")
    args = ap.parse_args()

    if args.reset_weak:
        n = _clear_weak_labels()
        print(f"已清除 {n} 条弱监督标签。")
        if args.status:
            return

    if args.status:
        print("── 标注样本状态 ──")
        print(label_summary())
        s = stats()
        need = needed_series()
        if need:
            print(f"\n待标注样本分布在 {len(need)} 个品种/周期：")
            for sym, tf, n in need:
                print(f"  · {sym} {tf}：{n} 条")
        if s["labeled"] >= 120:
            print(f"\n样本已够（{s['labeled']} ≥ 120），可以跑校准：")
            print("  python tools/calibrate_laya.py")
        else:
            print(f"\n样本不足（{s['labeled']} < 120），先造标签：")
            print("  python tools/weak_label.py")
        return

    print("── 弱监督自动标注 ──")
    print(label_summary())

    # 自动发现所有待标注样本涉及的 (symbol, timeframe)，按**样本时间范围**回溯拉取。
    # 样本天然横跨多个品种/周期，且可能来自数周前——只拉「最新 N 根」会大面积失配。
    spans = _sample_time_range()
    if not spans:
        print("没有待标注样本（或全部已标注）。")
        return
    if args.symbol:
        spans = {k: v for k, v in spans.items() if k[0] == args.symbol}
        if not spans:
            print(f"没有 {args.symbol} 的待标注样本。")
            return
    if args.timeframe_override:
        spans = {(s, args.timeframe_override): v for (s, _t), v in spans.items()}

    import time as _time

    now_ms = int(_time.time() * 1000)
    print(f"\n待标注样本分布在 {len(spans)} 个品种/周期组合，按样本时间范围回溯拉取：")
    frames: list[Any] = []
    for (sym, tf), (lo, hi) in sorted(spans.items(), key=lambda kv: -kv[1][0]):
        days = max(1, round((now_ms - lo) / 86_400_000))
        print(f"  · {sym} {tf}：样本跨度约 {days} 天 "
              f"({_fmt_ts(lo)} → {_fmt_ts(min(hi, now_ms))})", flush=True)
        got = _fetch_span(sym, tf, lo, now_ms)
        if got and got[0].bars:
            frames.extend(got)
            print(f"    取到 {len(got[0].bars)} 根", flush=True)
        else:
            print(f"    [跳过] {sym} {tf} 拉取失败或该时段无数据")
    if not frames:
        print("[错误] 所有品种都没拉到 K 线，终止。", file=sys.stderr)
        sys.exit(1)

    result = relabel_unlabeled(
        frames=frames, atr_mult=args.atr_mult, dry_run=args.dry_run
    )
    if "error" in result:
        print(f"[错误] {result['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"\n索引 {result['series']} 个品种序列，扫描 {result['scanned']} 条"
          f"｜已标注跳过 {result['already']}"
          f"｜本次标注 {result['labeled']}｜无法判定 {result['undecidable']}")

    # 失败原因细分——不细分的话用户只会看到"标不上"，不知道该做什么
    if result["undecidable"]:
        print("\n无法判定的原因分解：")
        if result["no_future"]:
            print(f"  · {result['no_future']} 条：决策时刻之后还没有 K 线"
                  f"（样本太新，等行情走出来再来标）")
        if result["gap_too_large"]:
            w = result.get("worst") or {}
            print(f"  · {result['gap_too_large']} 条：样本 close 与当前 MT5 行情"
                  f"对不上（缺口最大 {result['max_gap_atr_seen']:.1f} 倍 ATR）")
            if w:
                print(f"      最差样本：{w.get('symbol')} {w.get('timeframe')} "
                      f"close={w.get('close')} 但当前行情首根={w.get('fut_first')}")
            print("      原因：经纪商报价源/合约规格与样本产生时不同，"
                  "历史价格无法回溯对齐。")
            print("      这些样本只能靠人工在报告里「进入标注模式」补标。")
        if result["no_atr"]:
            print(f"  · {result['no_atr']} 条：样本缺 ATR，无法定阈值")

    if args.dry_run:
        print("\n[干跑] 未写文件。去掉 --dry-run 生效。")
    elif result["labeled"]:
        print(f"已回写 {len(result['files'])} 个文件。")
    else:
        print("\n没有新增标签。")

    print()
    print(label_summary())
    if stats()["labeled"] >= 120:
        print("\n下一步：python tools/calibrate_laya.py")
    else:
        print("\n继续积累：多跑几天，或在 Laya 报告里「进入标注模式」人工补标。")


if __name__ == "__main__":
    main()
