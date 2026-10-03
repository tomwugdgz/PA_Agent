# -*- coding: utf-8 -*-
"""弱监督自动标注：用未来真实走势当老师，判定 Laya 当时答对没有。

为什么需要它
------------
Laya 0.3.22 零样本在真实行情上置信度只有 ~0.14-0.17（作者原话 "near chance"）。
要让它的置信度变得可信，需要成百上千条**带正确答案**的样本。
手工标注每题要点 6 个下拉框——坚持不下来。

弱监督的做法：拿 Laya 当时给出的方向判断，等N 根 K 线之后看真实走势，
对/错就是天然的标签。**你什么都不用做，报告照常生成，样本自动积累。**

判定规则（只用可验证的客观量，不掺主观观点）
------------------------------------------
1. **周期结构**（先判，其余依赖它）：用未来窗口的 **Kaufman 效率比**
   ER = |净变动| / Σ|逐根变动|。ER ≥ 0.55 → ``trending_tr``（趋势）；
   ER ≤ 0.25 → ``trading_range``（区间）；中间灰区**不写标签**。
2. **方向**：趋势市里净涨跌幅 > +0.8·ATR → ``bullish``，< -0.8·ATR → ``bearish``，
   不足阈值 → ``neutral``。**震荡市（ER ≤ 0.25）一律给 ``neutral``**——
   区间内的短暂涨跌不是可交易方向，若按净涨跌硬判会产出「看涨 + 区间」这种
   自相矛盾的标签，拿去校准等于教模型胡猜。
3. **信号有效**：方向为 bullish/bearish → 有效，否则无效。

不可判定的样本（缺 K 线、缺 ATR、缺未来窗口、ER 落在灰区）一律跳过，
宁可少标也绝不写脏标签——脏标签会直接污染置信度校准。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from pa_agent.ai.laya_annotation import annotations_dir, _iter_samples

logger = logging.getLogger(__name__)

#: 净涨跌需超过多少倍 ATR 才算「有方向」
DEFAULT_ATR_MULT = 0.8
#: 效率比高于此值算趋势，低于 ER_RANGE 算区间
ER_TREND = 0.55
ER_RANGE = 0.25

#: 时间周期 → 毫秒。用于把「推理时刻」对齐到 K 线边界。
_TF_MS = {
    "M1": 60_000, "M3": 180_000, "M5": 300_000, "M15": 900_000,
    "M30": 1_800_000, "H1": 3_600_000, "H4": 14_400_000,
    "D1": 86_400_000, "W1": 604_800_000,
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000,
    "1d": 86_400_000, "1w": 604_800_000,
}


def _tf_ms(timeframe: str) -> int:
    """周期字符串→毫秒；未知周期按 15 分钟兜底。"""
    return _TF_MS.get(str(timeframe), 900_000)


def efficiency_ratio(closes: list[float]) -> float:
    """Kaufman 效率比 = |净变动| / Σ|逐根变动|。0=纯震荡, 1=单边直线。

    ``closes`` 必须是**连续的**价格序列（第一根就是起点）。若第一根与决策
    收盘价之间有缺口，务必把决策价插到序列头部再调用——否则净变动与路径
    长度不是同一段区间，算出的ER 可以 >1（数学上不可能，真实数据里确实
    出现过 3.27这种值）。
    """
    if len(closes) < 3:
        return 0.0
    net = abs(closes[-1] - closes[0])
    path = sum(abs(closes[i + 1] - closes[i]) for i in range(len(closes) - 1))
    if path <= 0:
        return 0.0
    return min(1.0, net / path)


def infer_labels(
    *,
    close: float,
    atr: float | None,
    future: list[float],
    atr_mult: float = DEFAULT_ATR_MULT,
    max_gap_atr: float = 1.5,
) -> dict[str, str] | None:
    """根据未来走势推断正确答案。无法判定时返回 None。

    Args:
        close: 决策时点的收盘价。
        atr: 当时的 ATR14；None 或 <=0 时无法定阈值，返回 None。
        future: 决策之后 N 根 K 线的收盘价（**时间正序**，[0] 是最近的一根）。
        atr_mult: 净涨跌需超过 ``atr_mult * atr`` 才算有方向。
        max_gap_atr: ``future[0]`` 与 ``close`` 的最大允许缺口（单位 ATR）。
            超过说明 K 线窗口没对齐到决策时刻，返回 None。
            短周期需放宽（1m 约 6、5m 约 4），调用方按周期传入。

    ⚠️ 剩余阻断项：XAUUSD 等少数品种经纪商更换过报价源 / 合约规格，
    历史 ``close`` 在当前 MT5 里**根本不存在**（实测样本 2001.06 vs 现价 4187.95，
    缺口 486 ATR；用 ±14 小时全偏移扫描 + 全历史精确匹配均找不到该价格）。
    这类样本不是时间对齐能修的，只能人工标注或等新样本。
    """
    if not future or len(future) < 2:
        return None
    try:
        c0 = float(close)
        a = float(atr) if atr else 0.0
        # 保持调用方传入的时间顺序（最早 → 最近）；只做数值清洗，不排序
        seq = [float(x) for x in future]
    except (TypeError, ValueError):
        return None
    if c0 <= 0 or a <= 0 or any(x <= 0 for x in seq):
        return None

    # 缺口守卫：决策价必须与未来第一根连续。缺口过大 = 数据没对齐，宁可丢弃。
    if abs(seq[0] - c0) > a * max_gap_atr:
        return None

    # 效率比必须在**含决策价**的连续序列上算，否则净变动与路径长度区间不一致
    er = efficiency_ratio([c0] + seq)

    net = seq[-1] - c0
    thresh = a * atr_mult

    labels: dict[str, str] = {}
    # 先判结构：效率比是「这段行情有没有方向」的第一性判据
    if er >= ER_TREND:
        labels["周期结构"] = "trending_tr"
    elif er <= ER_RANGE:
        labels["周期结构"] = "trading_range"
    # 0.25 < ER < 0.55 是「准趋势/准区间」灰区，结构题不写标签（宁可少标不可标错）

    # 再判方向。**震荡市一律给 neutral**——区间内的短暂涨跌不是可交易的方向，
    # 若还按净涨跌给 bullish，会产出「看涨 + 区间」这种自相矛盾的标签，
    # 拿去做校准等于教模型胡猜。
    if labels.get("周期结构") == "trading_range":
        labels["方向"] = "neutral"
    elif net > thresh:
        labels["方向"] = "bullish"
    elif net < -thresh:
        labels["方向"] = "bearish"
    else:
        # 趋势但净变动不足阈值：方向不明确，也算不可交易
        labels["方向"] = "neutral"

    labels["信号有效"] = "有效" if labels["方向"] in ("bullish", "bearish") else "无效"

    return labels or None


def _future_closes_for(sample: dict[str, Any],
                       bars_by_key: dict[tuple[str, str], list[tuple[int, float]]],
                       *,
                       lookahead: int = 20,
                       anchor_tol_atr: float = 2.0,
                       atr: float | None = None,
                       ) -> list[float]:
    """从 K 线库取出该样本决策时点**之后**的收盘价（时间正序）。

    定位分两级：
    1. **时间定位**：用 ``ts_ms`` 找到样本决策时刻所在/紧邻的那根；
    2. **价格校验**：样本 ``close`` 取自**已收盘的上一根**（不是决策时刻的实时价），
       所以时间上定位到的那根未必价格吻合。实测两者可差 1~2 根。
       因此在时间定位点附近小范围回退，直到 ``close`` 对得上（或退无可退）。

    ``anchor_tol_atr`` 是价格**回退匹配**的容差（单位 ATR），传 ``atr`` 才生效；
    不传则退化为按价格相对误差（1e-5）判断。精确匹配（相对误差 1e-9）不受它影响。
    短周期需要更宽的容差：实测 1m 品种相邻根价差可达 4~5 ATR（噪音远大于 15m+），
    故按周期自适应放大。
    ``lookahead`` 限制窗口长度：只看最近 N 根，太多会把信号稀释成长期趋势。
    """
    ctx = sample.get("context") or {}
    symbol = str(ctx.get("symbol") or "")
    tf = str(ctx.get("timeframe") or "")
    series = bars_by_key.get((symbol, tf))
    if not series:
        return []

    # 短周期噪音大，价格锚点要放宽（实测中位缺口：1m 约 4 ATR，15m 约 0 ATR）
    tf_tol_mult = {"1m": 6.0, "5m": 4.0}.get(tf, anchor_tol_atr)

    close = ctx.get("close")
    try:
        pivot = float(close) if close is not None else None
    except (TypeError, ValueError):
        pivot = None
    if pivot is not None and pivot <= 0:
        pivot = None

    # 价格容差：优先用 ATR 相对容差，缺失时退回价格相对误差
    tol: float | None = None
    if atr:
        try:
            a = float(atr)
            if a > 0:
                tol = a * max(0.05, tf_tol_mult)
        except (TypeError, ValueError):
            tol = None
    if tol is None and pivot is not None:
        tol = max(1e-9, abs(pivot) * 1e-5)

    def _close_enough(value: float) -> bool:
        return tol is not None and pivot is not None and abs(value - pivot) <= tol

    ts = sample.get("ts_ms")
    anchor = -1
    if ts is not None:
        try:
            cut = int(ts)
            bar_ms = _tf_ms(tf)
            # 决策时刻落在「当根已收盘 K 线」之内（ts_open <= cut < 下一根开盘）
            pos = 0
            for i, (t, _c) in enumerate(series):
                if t <= cut:
                    pos = i
                else:
                    break

            # 价格锚点分两阶段，**顺序不能反**：
            # 1) 先在时间锚点附近找**精确匹配**（相对误差 1e-9）。
            #    样本 close 来自已收盘 K 线，与时间锚点可差 1~5 根（实测 -5 根命中）。
            # 2) 精确匹配找不到，才退回 ATR 容差匹配。
            # 反过来会出错：ATR 容差（如 2×ATR）往往比相邻根的价差还宽，
            # 会把时间锚点本身误判为命中，从而取到错误位置的未来序列
            #（实测导致 |fut[0]-close| = 2.42 ATR，被缺口守卫全判为漂移）。
            if pivot is not None:
                exact_tol = max(1e-9, abs(pivot) * 1e-9)
                for delta in range(0, 9):
                    hit = -1
                    for cand in ((pos,) if delta == 0 else (pos - delta, pos + delta)):
                        if 0 <= cand < len(series) and abs(series[cand][1] - pivot) <= exact_tol:
                            hit = cand
                            break
                    if hit >= 0:
                        anchor = hit
                        break
            if anchor < 0 and tol is not None and pivot is not None:
                for delta in range(0, 9):
                    hit = -1
                    for cand in ((pos,) if delta == 0 else (pos - delta, pos + delta)):
                        if 0 <= cand < len(series) and _close_enough(series[cand][1]):
                            hit = cand
                            break
                    if hit >= 0:
                        anchor = hit
                        break
            if anchor < 0:
                # 没找到价格吻合点：退回纯时间锚点（让下游缺口守卫去判）
                anchor = pos
            if anchor + 1 < len(series) and bar_ms > 0:
                # 必须从锚点**紧接**往后取 lookahead 根，不能用 [-lookahead:]。
                # 后者从序列末尾截断：若锚点之后还有 100+ 根（样本太老、窗口很大），
                # 会跳过紧邻的走势、把「未来」错取到很远的位置，
                # 实测导致 |fut[0]-close| 高达 2.42 ATR 而被缺口守卫全判为漂移。
                tail = [c for _t, c in series[anchor + 1:]]
                return tail[:lookahead] if len(tail) > lookahead else tail
        except (TypeError, ValueError):
            pass

    if pivot is None:
        return []
    # 退化路径：找最后一个收盘价≈ pivot 的位置，取其后的一段
    last = -1
    for i, (_t, c) in enumerate(series):
        if abs(c - pivot) <= max(1e-9, abs(pivot) * 1e-6):
            last = i
    if last < 0 or last + 2 >= len(series):
        return []
    tail = [c for _t, c in series[last + 1:]]
    return tail[:lookahead] if len(tail) > lookahead else tail


def build_bar_index(frames: list[Any], *, horizon: int = 0) -> dict[tuple[str, str], list[tuple[int, float]]]:
    """把 frame 列表压成 ``(symbol, timeframe) -> [(ts_ms, close)]``（时间正序）。

    Args:
        horizon: 每组保留的最大根数。``0`` = **全部保留**（默认）。
            早先固定 20 会把几周前的样本全部挤出索引，导致弱监督覆盖率极低
            （实测 110 条只覆盖 9 条）。判定需要的是「决策时刻之后」的数据，
            所以必须保留覆盖到最老样本的整段历史。
    """
    index: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for f in frames:
        sym = str(getattr(f, "symbol", "") or "")
        tf = str(getattr(f, "timeframe", "") or "")
        bars = getattr(f, "bars", None) or []
        if not sym or not tf or not bars:
            continue
        rows: list[tuple[int, float]] = []
        for b in bars:
            # KlineBar 用ts_open（毫秒）
            ts = getattr(b, "ts_open", None)
            if ts is None:
                ts = getattr(b, "ts_ms", None)
            c = getattr(b, "close", None)
            if ts is None or c is None:
                continue
            try:
                ts_i = int(float(ts))
                # 兼容秒级时间戳
                if ts_i < 10_000_000_000:
                    ts_i *= 1000
                rows.append((ts_i, float(c)))
            except (TypeError, ValueError):
                continue
        if rows:
            rows = sorted(rows)
            index[(sym, tf)] = rows[-horizon:] if horizon > 0 else rows
    return index


def needed_series(experience_dir: Path | None = None) -> list[tuple[str, str, int]]:
    """统计未标注样本涉及哪些 ``(symbol, timeframe)`` 及各自条数。

    样本天然分散在多个品种/周期上（实测 110 条横跨 9 个组合）。只拉一个品种
    会让绝大多数样本因「找不到对应 K 线」而无法判定——必须按分布批量拉取。
    """
    from collections import Counter

    out_dir = annotations_dir(experience_dir)
    cnt: Counter[tuple[str, str]] = Counter()
    if not out_dir.is_dir():
        return []
    for p in sorted(out_dir.glob("*.jsonl")):
        try:
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        s = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if s.get("label"):
                        continue
                    ctx = s.get("context") or {}
                    sym, tf = ctx.get("symbol"), ctx.get("timeframe")
                    if sym and tf:
                        cnt[(str(sym), str(tf))] += 1
        except OSError:
            continue
    return [(s, t, n) for (s, t), n in cnt.most_common()]


def relabel_unlabeled(
    *,
    frames: list[Any],
    experience_dir: Path | None = None,
    atr_mult: float = DEFAULT_ATR_MULT,
    max_gap_atr: float = 1.5,
    dry_run: bool = False,
) -> dict[str, Any]:
    """扫描未标注样本，用弱监督回填 ``label``。返回统计（含失败原因细分）。

    幂等：已标注的行会被跳过，重复运行不会重复写入。
    """
    out_dir = annotations_dir(experience_dir)
    result = {"scanned": 0, "already": 0, "labeled": 0, "undecidable": 0,
              "files": [], "series": 0,
              # 失败原因细分，用于给用户可行动的诊断
              "no_future": 0, "gap_too_large": 0, "no_atr": 0,
              "max_gap_atr_seen": 0.0, "worst": None}
    if not out_dir.is_dir():
        return result
    bar_index = build_bar_index(frames)
    result["series"] = len(bar_index)
    if not bar_index:
        result["error"] = "没有可用 K 线，无法判定未来走势"
        return result

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
            result["scanned"] += 1
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            if s.get("label"):
                result["already"] += 1
                continue
            ctx = s.get("context") or {}
            # 缺口守卫的阈值要跟锚点容差同源：1m 品种相邻根价差可达 4~5 ATR，
            # 用统一 1.5 ATR 会把已对齐的样本也误判为漂移。
            tfm = str(ctx.get("timeframe") or "")
            gap_mult = max(max_gap_atr, {"1m": 6.0, "5m": 4.0}.get(tfm, max_gap_atr))
            fut = _future_closes_for(s, bar_index, atr=ctx.get("atr"))
            labels = infer_labels(
                close=ctx.get("close"),
                atr=ctx.get("atr"),
                future=fut,
                atr_mult=atr_mult,
                max_gap_atr=gap_mult,
            )
            if not labels:
                result["undecidable"] += 1
                # 细分原因
                if not fut:
                    result["no_future"] += 1
                    continue
                try:
                    a = float(ctx.get("atr") or 0)
                    c0 = float(ctx.get("close") or 0)
                    gap = abs(float(fut[0]) - c0) / a if a > 0 else float("inf")
                except (TypeError, ValueError, ZeroDivisionError):
                    gap = float("inf")
                if a <= 0:
                    result["no_atr"] += 1
                else:
                    result["gap_too_large"] += 1
                    if gap > result["max_gap_atr_seen"]:
                        result["max_gap_atr_seen"] = gap
                        result["worst"] = {
                            "symbol": ctx.get("symbol"),
                            "timeframe": ctx.get("timeframe"),
                            "close": c0,
                            "atr": a,
                            "fut_first": float(fut[0]),
                            "gap_atr": gap,
                        }
                continue
            s["label"] = labels
            s["source"] = "weak_supervision"
            s["labeled_at_ms"] = int(time.time() * 1000)
            s["label_method"] = f"future_{len(fut)}bars_atr{atr_mult}"
            lines[i] = json.dumps(s, ensure_ascii=False)
            changed += 1
            result["labeled"] += 1
        if changed and not dry_run:
            try:
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                result["files"].append(str(path))
            except OSError as exc:
                logger.warning("回写标注失败 %s: %s", path, exc)
    return result


def label_summary(experience_dir: Path | None = None) -> str:
    """人类可读的样本概况，供 GUI / CLI 展示。"""
    from pa_agent.ai.laya_annotation import stats

    s = stats(experience_dir)
    return (
        f"样本 {s['total']} 条｜已标注 {s['labeled']}"
        f"（人工 {s['manual']} / 弱监督 {s['weak']}）｜待标注 {s['unlabeled']}"
    )


def list_unlabeled(experience_dir: Path | None = None, limit: int = 10) -> list[dict[str, Any]]:
    """列出待标注样本摘要（调试 / GUI 提示用）。"""
    out_dir = annotations_dir(experience_dir)
    rows: list[dict[str, Any]] = []
    if not out_dir.is_dir():
        return rows
    for s in _iter_samples(out_dir):
        if s.get("label"):
            continue
        ctx = s.get("context") or {}
        rows.append({
            "symbol": ctx.get("symbol"),
            "timeframe": ctx.get("timeframe"),
            "close": ctx.get("close"),
            "ts_ms": s.get("ts_ms"),
        })
        if len(rows) >= limit:
            break
    return rows
