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
    test_target_vectors()
    test_to_laya_pairs_priority()
    test_confidence_filter()
    test_calibration_path_is_next_to_weights()
    print("全部通过 ✅")


if __name__ == "__main__":
    main()
