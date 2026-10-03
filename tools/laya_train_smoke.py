# -*- coding: utf-8 -*-
"""Laya 板块冒烟：弱监督标注 + 校准数据转换（不连 MT5、不加载 Laya 权重）。

用法：python tools/laya_train_smoke.py
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pa_agent.ai.laya_annotation import (  # noqa: E402
    _target_vector,
    to_laya_pairs,
)
from pa_agent.ai.laya_schema import build_questions  # noqa: E402
from pa_agent.ai.laya_weaklabel import (  # noqa: E402
    _future_closes_for,
    build_bar_index,
    efficiency_ratio,
    infer_labels,
)
from pa_agent.data.base import KlineBar  # noqa: E402


def test_efficiency_ratio() -> None:
    """ER：单边→1.0，纯震荡→<0.25。"""
    assert abs(efficiency_ratio([1, 1.1, 1.2, 1.3, 1.4]) - 1.0) < 1e-9
    assert efficiency_ratio([1, 1.2, 1.0, 1.25, 1.05]) < 0.25
    assert 0.25 < efficiency_ratio(
        [100, 101.5, 100.8, 102.3, 101.6, 103.1, 102.4, 103.9, 103.2, 104.7, 104.0, 105.5]
    ) < 0.55, "灰区样本 ER 应落在 0.25~0.55"
    print("  [OK] Kaufman 效率比：单边 1.0 / 震荡 0.06 / 灰区 0.44")


def test_infer_labels_trend() -> None:
    """趋势市：净涨跌超阈值 → bullish/bearish + trending_tr。"""
    up = infer_labels(close=100, atr=1, future=[100.2 + i * 0.25 for i in range(20)])
    assert up == {"方向": "bullish", "信号有效": "有效", "周期结构": "trending_tr"}, up
    dn = infer_labels(close=100, atr=1, future=[99.8 - i * 0.25 for i in range(20)])
    assert dn == {"方向": "bearish", "信号有效": "有效", "周期结构": "trending_tr"}, dn
    print("  [OK] 趋势市：上涨→bullish、下跌→bearish，结构均trending_tr")


def test_efficiency_ratio_cannot_exceed_one() -> None:
    """ER 必须 <= 1。缺口序列曾算出 3.27这种数学上不可能的值。"""
    gap = efficiency_ratio([157.865, 157.876, 157.861, 157.87, 157.87, 157.862])
    assert 0.0 <= gap <= 1.0, f"ER 必须落在 [0,1]，实得 {gap}"
    # 决策价接到头部后，ER 应显著更高（这才反映真实单边程度）
    with_head = efficiency_ratio([158.317] + [157.865, 157.876, 157.861, 157.87, 157.87, 157.862])
    assert with_head > gap, (gap, with_head)
    print(f"  [OK] ER 上界：无头部 {gap:.3f} / 含决策价头部 {with_head:.3f}，均 <= 1")


def test_infer_labels_choppy_overrides_direction() -> None:
    """关键回归：震荡市即使末端上涨，也必须给 neutral（不能产出「看涨+区间」矛盾标签）。"""
    # 序列必须「连续」：第一根紧邻 close，否则缺口守卫会先把它丢掉。
    fut = [100.0] + [100 + 3 * math.sin(i * 1.7) for i in range(1, 21)]
    net = fut[-1] - 100
    assert net > 0.8, f"前置条件：净涨 {net:.2f} 必须超过 0.8×ATR，否则测不到保护逻辑"
    assert abs(fut[0] - 100) < 1.5, "前置条件：fut[0] 必须紧邻 close，否则被缺口守卫丢掉"
    lab = infer_labels(close=100, atr=1, future=fut)
    assert lab is not None, f"不应被缺口守卫丢掉：{lab}"
    assert lab["周期结构"] == "trading_range", lab
    assert lab["方向"] == "neutral", f"震荡市不该给方向信号，实得 {lab}"
    assert lab["信号有效"] == "无效", lab
    print(f"  [OK] 震荡保护：净涨 {net:.2f} > 0.8×ATR 仍判neutral（结构/方向不矛盾）")


def test_gap_guard_rejects_misaligned() -> None:
    """缺口守卫：样本 close 与当前行情差太远（换报价源/错位）时必须拒绝标注。

    实测踩过这个坑：XAUUSD 样本 close=2001、当前 MT5 报价 4259，缺口 478 倍
    ATR，若不拦就会用完全错位的行情给样本贴标签，直接污染校准。
    """
    fut = [2000.0, 2001.0, 2002.0, 2003.0]     # 与 close=100 完全不同价位
    assert infer_labels(close=100, atr=1, future=fut) is None
    # 连续的小缺口仍应放行（0.5 ATR 在容忍范围内）
    ok = infer_labels(close=100, atr=1, future=[100.5, 101.2, 102.0, 103.1])
    assert ok is not None, "小缺口不应被误杀"
    print("  [OK] 缺口守卫：478×ATR 的错位样本被拒，0.5×ATR 的正常样本放行")


def test_infer_labels_grey_zone() -> None:
    """ER 灰区（0.25~0.55）不写结构标签（宁可少标不可标错）。"""
    seq = [100, 101.5, 100.8, 102.3, 101.6, 103.1, 102.4, 103.9, 103.2, 104.7, 104.0, 105.5]
    # ER 必须在「含决策价」的连续序列上算
    er = efficiency_ratio(seq)
    assert 0.25 < er < 0.55, f"前置条件：ER 应落在灰区，实得 {er:.3f}"
    assert abs(seq[0] - 100) < 1e-9, "前置条件：seq 起点须等于 close"
    lab = infer_labels(close=100, atr=1, future=seq[1:])
    assert lab is not None, f"不该被缺口守卫丢掉：{lab}"
    assert "周期结构" not in lab, f"灰区不该写结构标签，实得 {lab}"
    print(f"  [OK] ER 灰区（{er:.2f}）样本不写结构标签")


def test_infer_labels_boundaries() -> None:
    """缺 ATR / 空未来 / 零 ATR 一律返回 None（不写脏标签）。"""
    assert infer_labels(close=100, atr=None, future=[1, 2, 3]) is None
    assert infer_labels(close=100, atr=0, future=[1, 2, 3]) is None
    assert infer_labels(close=100, atr=1, future=[]) is None
    assert infer_labels(close=0, atr=1, future=[1, 2, 3]) is None
    print("  [OK] 不可判定样本全部返回 None")


def test_future_slice() -> None:
    """未来切片：ts_ms 是推理时刻，需对齐到 K 线边界后取其后N 根。"""
    bars = [
        KlineBar(seq=i + 1, ts_open=1_700_000_000_000 + i * 900_000, open=1, high=1,
                 low=1, close=100 + i * 0.1, volume=1)
        for i in range(30)
    ]
    bars.sort(key=lambda b: b.seq, reverse=True)      # newest-first（项目约定）
    f = type("F", (), {})()
    f.symbol, f.timeframe, f.bars = "EURUSD", "15m", bars
    idx = build_bar_index([f], horizon=20)
    assert ("EURUSD", "15m") in idx
    cut = 1_700_000_000_000 + 20 * 900_000 + 100_000  # 第 21 根之中
    s = {"ts_ms": cut, "context": {"symbol": "EURUSD", "timeframe": "15m",
                                   "close": bars[20].close}}
    fut = _future_closes_for(s, idx)
    assert 8 <= len(fut) <= 10, f"应取到第 22~30 根共 9 根，实得 {len(fut)}"
    assert fut == sorted(fut), "未来序列必须时间正序"
    print(f"  [OK] 未来切片：决策点之后取到 {len(fut)} 根，时间正序")


def test_future_slice_takes_adjacent_bars() -> None:
    """回归：必须取锚点**紧接**的 N 根，不能用 ``[-lookahead:]`` 从末尾截断。

    原bug：``series[anchor + 1:][-lookahead:]`` 在锚点后还有 100+ 根时，
    会跳过紧邻走势、把「未来」错取到很远位置（实测 |fut[0]-close| 达 2.42 ATR，
    导致 105 条样本里 104 条被缺口守卫误判为报价漂移）。
    """
    from pa_agent.ai.laya_weaklabel import _future_closes_for as _ffs

    # 时间锚点落在 index 100（ts_ms 落在第 101 根之中），
    # 而锚点之后还剩 99 根 —— 足以暴露「从末尾截断」的 bug
    series = [(1_000 * i, 100.0 + i) for i in range(200)]
    s = {"ts_ms": 100_500, "context": {"symbol": "X", "timeframe": "1m",
                                       "close": 200.0, "atr": 1.0}}
    fut = _ffs(s, {("X", "1m"): series}, atr=1.0)
    # 精确匹配 anchor=100（close=200.0），紧接的 20 根 = 201..220
    assert fut[:3] == [201.0, 202.0, 203.0], f"应从锚点紧接处取，实得 {fut[:3]}"
    assert len(fut) == 20, f"应取 20 根，实得 {len(fut)}"
    assert fut[-1] == 220.0, f"末根应为 220.0（紧接窗口），实得 {fut[-1]}"
    # 若误用 [-lookahead:]，会拿到 180..199
    assert fut[0] != 180.0, "不应从序列末尾截断"
    print("  [OK] 未来切片取紧邻窗口：锚点后仍有 99 根，仍取紧接的 20 根")


def test_anchor_prefers_exact_over_tolerance() -> None:
    """回归：锚点定位必须**优先精确匹配**，不能先用 ATR 容差。

    原 bug：ATR 容差（2×ATR）比相邻根价差还宽，会把时间锚点本身误判为命中，
    从而取到错误位置的未来序列。实测该场景让 |fut[0]-close| 达 2.42 ATR。
    """
    from pa_agent.ai.laya_weaklabel import _future_closes_for as _ffs

    # 时间锚点落在 index 10 之内（ts_ms=10500 → pos=10，close=110）
    # 110 与 pivot=105 差 5.0 = 10×ATR，**落在 ATR 容差内**，
    # 但真正的精确匹配在 index 5（close=105.0）
    series = [(1_000 * i, 100.0 + i) for i in range(40)]
    s = {"ts_ms": 10_500, "context": {"symbol": "X", "timeframe": "1m",
                                       "close": 105.0, "atr": 5.0}}
    fut = _ffs(s, {("X", "1m"): series}, atr=5.0)
    # 精确匹配 anchor=5，应从 106.0 开始（而非容差匹配到的 111.0）
    assert fut[0] == 106.0, f"精确匹配后应从 106.0 起，实得 {fut[0]}"
    print("  [OK] 锚点优先精确匹配：容差内的非精确点未被误用")


def test_short_timeframe_tolerance_is_wider() -> None:
    """回归：短周期容差需比长周期宽（1m 噪音远大于 15m）。"""
    from pa_agent.ai.laya_weaklabel import _future_closes_for as _ffs

    # 1m：anchor 时间位置 close差 4 ATR，应仍能通过容差回退匹配
    series_1m = [(60_000 * i, 100.0) for i in range(30)]
    series_1m[20] = (60_000 * 20, 100.0)
    s = {"ts_ms": 20 * 60_000 + 1_000, "context": {"symbol": "X", "timeframe": "1m",
                                                    "close": 104.0, "atr": 1.0}}
    fut = _ffs(s, {("X", "1m"): series_1m}, atr=1.0)
    assert fut, "1m 短周期应能通过放宽的容差找到锚点"
    print(f"  [OK] 短周期容差自适应：1m 品种 4 ATR 偏差仍可定位（取到 {len(fut)} 根）")


def test_target_vectors() -> None:
    """one-hot 构造：choice 用 criteria 键序，noul 用 [1-P, P]。"""
    qs = build_questions()
    crit = qs["周期结构"]["criteria"]
    vec = _target_vector(qs["周期结构"], "broad_channel")
    keys = list(crit.keys())
    assert vec is not None and len(vec) == len(keys)
    assert vec[keys.index("broad_channel")] == 1.0 and sum(vec) == 1.0
    # 中文标签也能反查
    assert _target_vector(qs["周期结构"], "宽通道") == vec
    # noul
    assert _target_vector(qs["信号有效"], "有效") == [0.0, 1.0]
    assert _target_vector(qs["信号有效"], "无效") == [1.0, 0.0]
    # 未知答案返回 None（该题跳过，不污染数据集）
    assert _target_vector(qs["周期结构"], "不存在") is None
    print(f"  [OK] one-hot 构造：{len(keys)} 维周期结构、2 维 noul、未知答案返回 None")


def test_to_laya_pairs_priority() -> None:
    """人工标签优先级高于弱监督；无标签/坏样本被正确剔除。"""
    qs = build_questions()
    base = {"state": "x", "questions": qs, "context": {}}
    samples = [
        {**base, "label": {"方向": "bullish"}, "answers": {}, "source": "weak_supervision"},
        {**base, "label": {"方向": "bearish"}, "answers": {}, "source": "manual_label"},
        {**base, "label": None, "answers": {}, "source": "auto"},
        {**base, "label": {"不存在题": "x"}, "answers": {}, "source": "auto"},
        {**base, "label": {"方向": "bogus"}, "answers": {}, "source": "auto"},
    ]
    pairs, info = to_laya_pairs(samples)
    assert info["total"] == 5
    assert info["paired"] == 2, info
    assert info["dropped_no_label"] == 1, info
    assert info["dropped_bad_target"] == 2, info   # 不存在的题 + 未知选项值
    # 人工标签应排在前面
    first = pairs[0][2]["方向"]
    keys = list(qs["方向"]["criteria"].keys())
    assert first[keys.index("bearish")] == 1.0, f"人工标签未优先：{first}"
    print(f"  [OK] 样本筛选：5 条 → 2 条可用，人工标签优先，坏样本全部剔除")


def test_confidence_filter() -> None:
    """min_answer_confidence 能挡掉「模型自己都不知道」的样本。"""
    qs = build_questions()
    base = {"state": "x", "questions": qs, "label": {"方向": "bullish"}}
    low = {**base, "answers": {"方向": {"confidence": 0.10}}}
    high = {**base, "answers": {"方向": {"confidence": 0.80}}}
    _, info = to_laya_pairs([low, high], min_answer_confidence=0.35)
    assert info["paired"] == 1, info
    assert info["dropped_low_conf"] == 1, info
    print("  [OK] 低置信样本过滤：0.10 被丢、0.80 保留")


def test_calibration_path_is_next_to_weights() -> None:
    """校准文件放权重目录旁——换模型目录自动失效。"""
    from pa_agent.ai.laya_annotation import calibration_out_path, load_calibration_if_any

    p = calibration_out_path("/tmp/laya-models/laya")
    assert p.name == "calibration.json" and p.parent.name == "laya-models"
    assert load_calibration_if_any("/nonexistent/dir") is None
    print(f"  [OK] 校准路径：{p}")


def test_report_is_data_driven() -> None:
    """报告必须随导入数据变化——防止将来出现"固定数据出固定报告"的回归。

    用两批趋势相反的合成 K 线（单边涨 / 单边跌）各跑一次真实推理，
    断言喂给模型的状态文本与 Laya 答案都随之改变。
    """
    import math

    from pa_agent.data.base import KlineBar
    from pa_agent.data.snapshot import build_analysis_frame

    def make(direction: str, n: int = 120, start: float = 1.1000):
        bars, price = [], start
        drift = 0.0009 if direction == "up" else -0.0009
        for i in range(n):
            wave = 0.00035 * math.sin(i / 3.0)
            o = price
            c = price + drift + wave * 0.5
            bars.append(
                KlineBar(
                    seq=n - i,
                    ts_open=1_700_000_000_000 - i * 60_000,
                    open=round(o, 5),
                    high=round(max(o, c) + 0.0006, 5),
                    low=round(min(o, c) - 0.0006, 5),
                    close=round(c, 5),
                    volume=float(1000 + (i % 7) * 30),
                    closed=True,
                )
            )
            price = c
        return list(reversed(bars))  # bars[0] 必须是最新

    def state_of(direction: str) -> str:
        from pa_agent.report.laya_pipeline import generate_report
        from pa_agent.config.settings import Settings

        bars = make(direction)
        frame = build_analysis_frame(
            bars, len(bars), "TESTPAIR", "M15", now_ms=1_700_000_000_000
        )
        assert frame is not None and frame.bars, "frame 构造失败"
        return generate_report(frame, Settings()).state_text

    up, down = state_of("up"), state_of("down")
    assert up != down, "两批不同数据竟生成完全相同的状态文本——报告未使用导入数据"
    # 状态文本必须体现各自的区间边界（数据确实进了state）
    assert "1.17181" in up and "0.9922" in down, "状态文本未反映各自数据的区间"
    print("  [OK] 报告随数据变化：涨/跌两批数据状态文本不同，区间边界各自独立")


def test_branch_derivation() -> None:
    """三分支推导：恰好 3 个、多空概率可区分、虚高 RR 被惩罚。

    防三类回归：
      1. 分支数不等于 3（少一个方向 / 排序错乱）
      2. 多空概率完全相同（未按 bull/bear 分布拆分）
      3. 虚高 RR（>10）反而被当成"达标"加分
    """
    from types import SimpleNamespace

    from pa_agent.config.settings import Settings
    from pa_agent.report.laya_branches import (
        _direction_probs,
        _structure_factor,
        derive_branches,
    )

    cfg = Settings().laya

    # ── 方向拆分：多空必须能区分 ──
    answers = {
        "方向": SimpleNamespace(
            value="bearish", confidence=0.15,
            probabilities={"bullish": 0.31, "bearish": 0.57, "neutral": 0.12},
        ),
        "信号有效": SimpleNamespace(value=0.68, confidence=0.68, probabilities={}),
        "噪音区": SimpleNamespace(value=0.67, confidence=0.67, probabilities={}),
        "尺度冲突": SimpleNamespace(value=0.62, confidence=0.62, probabilities={}),
        "突破质量": SimpleNamespace(value="close_breakout", confidence=0.32, probabilities={}),
    }
    bull, bear = _direction_probs(answers)
    assert bear > bull, f"看空应大于看多（0.57 vs 0.31），实得 bull={bull:.4f} bear={bear:.4f}"
    assert bull + bear < 0.31, "拆分后绝对值应被校准压缩，不能等于原始 0.31"

    # ── 虚高 RR 必须被惩罚 ──
    f_ok, _ = _structure_factor(
        direction="long", has_structure=True, rr=2.0, risk_in_atr=1.2, actionable=True
    )
    f_bad, r_bad = _structure_factor(
        direction="long", has_structure=True, rr=91.0, risk_in_atr=0.35, actionable=True
    )
    assert f_bad < f_ok, f"虚高 RR+过近止损应被严惩：bad={f_bad:.2f} ok={f_ok:.2f}"
    assert any("虚高" in x for x in r_bad), "扣分明细必须指出 RR 虚高"
    assert any("统计意义" in x for x in r_bad), "扣分明细必须指出 R 无统计意义"

    # ── 端到端：恰好 3 个分支且按概率降序 ──
    features = SimpleNamespace(
        supports=(1.14788,), resistances=(1.17327,),
        range_high=1.2094, range_low=1.14788,
        invalidation_long=1.14788, invalidation_short=1.17327,
        measured_moves=(), swing_structure="insufficient",
    )
    branches = derive_branches(
        close=1.17277, atr=0.001582, features=features, cfg=cfg, answers=answers
    )
    assert len(branches) == 3, f"应恰好 3 个分支，实得 {len(branches)}"
    keys = {b.key for b in branches}
    assert keys == {"long", "short", "wait"}, f"分支 key 应为 多/空/观望，实得 {keys}"
    probs = [b.probability for b in branches]
    assert probs == sorted(probs, reverse=True), f"应按概率降序，实得 {probs}"
    lng = [b for b in branches if b.key == "long"][0]
    sht = [b for b in branches if b.key == "short"][0]
    # 防"下限钳制掩盖真实差异"：两侧都远低于 0.10 时仍必须可区分
    assert sht.probability > lng.probability, (
        f"看空应严格高于看多（bearish 0.57 > bullish 0.31），"
        f"实得 long={lng.probability:.4f} short={sht.probability:.4f}"
    )
    assert lng.low_prob and sht.low_prob, "两侧均 <10% 时都应标记为低概率"
    wait = [b for b in branches if b.key == "wait"][0]
    assert not wait.actionable, "观望分支绝不能标记为可执行"
    print(f"  [OK] 三分支推导：3 个分支（多/空/观望）、按概率降序、"
          f"多空可区分（{lng.probability:.0%} vs {sht.probability:.0%}）、"
          f"低概率已标注、虚高 RR 被惩罚")


def main() -> None:
    print("── Laya 板块冒烟（弱监督 + 校准数据）──")
    test_efficiency_ratio()
    test_efficiency_ratio_cannot_exceed_one()
    test_infer_labels_trend()
    test_infer_labels_choppy_overrides_direction()
    test_gap_guard_rejects_misaligned()
    test_infer_labels_grey_zone()
    test_infer_labels_boundaries()
    test_future_slice()
    test_future_slice_takes_adjacent_bars()
    test_anchor_prefers_exact_over_tolerance()
    test_short_timeframe_tolerance_is_wider()
    test_target_vectors()
    test_to_laya_pairs_priority()
    test_confidence_filter()
    test_calibration_path_is_next_to_weights()
    test_report_is_data_driven()
    test_branch_derivation()
    print("全部通过 ✅")


if __name__ == "__main__":
    main()
