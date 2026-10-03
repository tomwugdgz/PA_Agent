# -*- coding: utf-8 -*-
"""Laya 的问题集与市场状态构造。

设计约束（来自实测，见 tools/laya_smoke_test.py）
------------------------------------------------
Laya multilingual 档：`max_len=1024`（state 上限）、`head_max_len=256`
（**所有问题的选项文本共享**的预算）。中文在该 tokenizer 下约 **1 token/字**，
因此：

* 选项标签必须**极短**（≤8 字）。语义注解放到 state 里，不占 head 预算。
* state 控制在 ~700 字以内，避免截断。
* 选择题选项数必须 < 20（>20 官方文档明示会崩）。

Laya 不是生成式模型，不产出文本；本模块只定义三类原语：
choice（多选一 + 概率）、score（有序等级）、noul（校准后的 P(true)）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pa_agent.ai.cycle_enums import CYCLE_ORDER, CYCLE_POSITION_ZH


# ── 选择题 low-cost 标签 ─────────────────────────────────────────────────────
# 说明为何不直接用 CYCLE_POSITION_ZH：那里的标签带"（Spike）"等装饰，多占 token。
# 这里的标签刻意压缩到 2–5 字，把 head 预算留给更多问题。
_CYCLE_LABELS: dict[str, str] = {
    "spike": "急速尖峰",
    "micro_channel": "微型通道",
    "tight_channel": "窄通道",
    "normal_channel": "正常通道",
    "broad_channel": "宽通道",
    "trending_tr": "趋势型区间",
    "trading_range": "交易区间",
    "extreme_tr": "极端区间",
}

#: 语义注解，随 state 一并发送（不占 head 预算）
_CYCLE_GLOSSARY = (
    "结构词典：急速尖峰=连续同向几乎不重叠；微型通道=极窄紧密推进；"
    "窄通道=回撤浅重叠多；正常通道=推进回撤均衡；宽通道=回撤深波动大；"
    "趋势型区间=区间内有方向倾斜；交易区间=上下边界明确方向中性；"
    "极端区间=边界极宽噪音大。"
)


def build_cycle_criteria() -> dict[str, str]:
    """返回 cycle-choice 的 {enum: 短标签}，顺序与 CYCLE_ORDER 一致。

    只保留 CYCLE_ORDER 中存在且有短标签的项，防止 enum 变更后答案为空。
    """
    out: dict[str, str] = {}
    for key in CYCLE_ORDER:
        label = _CYCLE_LABELS.get(key) or CYCLE_POSITION_ZH.get(key) or key
        out[key] = label
    return out


#: 突破质量枚举，与 market_features.SimpleMarketFeatures.breakout_quality 对齐
_BREAKOUT_QUALITY_LABELS: dict[str, str] = {
    "none": "无突破",
    "wick_probe": "影线试探",
    "close_breakout": "收盘突破",
    "failed": "突破失败",
    "testing": "回测中",
    "surviving": "站稳存续",
}


def build_questions(*, with_breakout: bool = True) -> dict[str, dict[str, Any]]:
    """构造 Laya 问题集。

    Args:
        with_breakout: 是否在报告中包含突破质量 / 噪音类问题。
            关闭后只剩 3 个核心问题，head 预算最宽松（用于低配或排障）。
    """
    questions: dict[str, dict[str, Any]] = {
        "周期结构": {
            "type": "choice",
            "instructions": "这段行情属于哪种市场周期结构？",
            "criteria": build_cycle_criteria(),
        },
        "方向": {
            "type": "choice",
            "instructions": "当前多空倾向？",
            "criteria": {
                "bullish": "偏多",
                "bearish": "偏空",
                "neutral": "中性",
            },
        },
        "信号有效": {
            "type": "noul",
            "instructions": "当前是否存在足够清晰的交易信号？",
        },
    }
    if with_breakout:
        questions["突破质量"] = {
            "type": "choice",
            "instructions": "最近一次边界突破处于什么状态？",
            "criteria": dict(_BREAKOUT_QUALITY_LABELS),
        }
        questions["噪音区"] = {
            "type": "noul",
            "instructions": "当前是否处于难以交易的噪音区？",
        }
        questions["尺度冲突"] = {
            "type": "noul",
            "instructions": "长周期与短周期方向是否互相矛盾？",
        }
    return questions


# ── 结果容器 ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LayaAnswer:
    """单个问题的答案（无论 choice / noul / score）。"""

    qid: str
    kind: str                      # choice | noul | score
    value: Any                     # choice->str, noul->float(P true), score->str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    #: 置信度是否跨过门槛；False 时下游不得据此生成价格建议
    reliable: bool = True


@dataclass(frozen=True)
class LayaPrediction:
    """一次 Laya 推理的完整结果。"""

    answers: dict[str, LayaAnswer]
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    #: 加载耗时（首次）与设备，用于报告页脚说明
    load_ms: float = 0.0
    #: 是否挂载了置信度校准（calibration.json）。False = 零样本原始置信度
    calibrated: bool = False
    device: str = ""

    def get(self, qid: str) -> LayaAnswer | None:
        return self.answers.get(qid)


# ── state 构造 ───────────────────────────────────────────────────────────────

# state 目标 ≤ ~700 字；超出会被截断（max_len=1024，实测 1 token/中文字）
_STATE_CHAR_BUDGET = 700


def build_state(
    *,
    symbol: str,
    timeframe: str,
    features: Any,
    atr: float | None,
    close: float,
) -> dict[str, Any]:
    """把预先算好的 SimpleMarketFeatures 压成一段紧凑中文状态文本。

    刻意**不**把 OHLC 原始数据喂进去：Laya 的语义空间是文本叙述，
    而 position / ATR 距离 / 比率这类**定量关系**它算不准
    （官方文档：算术/计数/日期比较一律走确定性代码）。
    """
    parts: list[str] = [f"品种 {symbol}，周期 {timeframe}。"]

    # ── 区间与位置
    if features.range_high is not None and features.range_low is not None:
        parts.append(
            f"近 {features.lookback_bars} 根区间 {features.range_low}~{features.range_high}，"
            f"宽度 {features.range_width_atr} 倍 ATR。"
        )
    parts.append(f"现价 {close}，位于区间 {features.price_position} 分位（{features.zone}）。")
    if features.dist_to_high_atr is not None and features.dist_to_low_atr is not None:
        parts.append(
            f"距上沿 {features.dist_to_high_atr} 倍 ATR，距下沿 {features.dist_to_low_atr} 倍 ATR。"
        )
    if atr:
        parts.append(f"ATR14 = {atr}。")

    # ── 形态与摆动
    parts.append(f"摆动结构：{features.swing_structure}。")
    if features.pullback_depth_atr is not None:
        parts.append(
            f"回撤深度 {features.pullback_depth_atr} 倍 ATR，"
            f"已持续 {features.pullback_bars} 根。"
        )

    # ── 重叠 / 噪音（铁丝网）
    if features.overlap_mean_10 is not None:
        parts.append(f"近 10 根重叠率 {features.overlap_mean_10}，")
        parts.append(f"十字星内包占比 {features.doji_inside_ratio_10}，")
        parts.append(
            f"铁丝网评分 {features.barbwire_score}"
            f"（{'是' if features.barbwire_candidate else '否'}噪音候选）。"
        )

    # ── H/L 计数：这是纯粹的计数事实，模型无法自行推导，必须显式给
    hc = features.hl_count
    parts.append(
        f"H/L 计数：多头 {hc.bull_count}（{hc.bull_candidate}），"
        f"空头 {hc.bear_count}（{hc.bear_candidate}）。"
    )

    # ── 突破与多尺度
    quality_zh = _BREAKOUT_QUALITY_LABELS.get(
        features.breakout_quality, features.breakout_quality
    )
    parts.append(
        f"突破质量：{quality_zh}"
        f"（尝试类型 {features.breakout_attempt_type or '无'}）。"
    )
    parts.append(
        f"余波提示：{features.spike_aftermath_hint}；"
        f"多尺度方向冲突：{'是' if features.scale_conflict else '否'}。"
    )

    # ── 结构位（让模型知道边界在哪，但不让它算价格）
    if features.supports:
        parts.append(
            "支撑：" + "、".join(str(p) for p in features.supports[:3]) + "。"
        )
    if features.resistances:
        parts.append(
            "阻力：" + "、".join(str(p) for p in features.resistances[:3]) + "。"
        )
    if features.measured_moves:
        mm = features.measured_moves[0]
        parts.append(f"测算目标：{mm.kind} 高度 {mm.height}，投影 {mm.target_price}。")

    state = "".join(parts)
    if len(state) > _STATE_CHAR_BUDGET:
        state = state[:_STATE_CHAR_BUDGET]
    # 词典放最后：即使被截断，丢的也是词典而不是行情事实
    return {"body": _CYCLE_GLOSSARY + state}
