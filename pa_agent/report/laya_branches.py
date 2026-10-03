"""Laya 分支推导：同一份数据推出三个交易判断（多/空/观望）并给出复合概率。

设计要点（为什么不能直接用 Laya 的 probabilities）
----------------------------------------------------
Laya 输出的 ``LayaAnswer.probabilities`` 是**未校准的原始概率**，而
``LayaAnswer.confidence`` 是**校准后**置信度。实测方向题原始
``bearish=0.5725``，校准后仅 ``0.1513``——差1.94 倍（温度系数）。

直接按原始概率排序取"概率最高"，会报出57% 这类数字，而它的可信度
接近随机。因此本模块采用**复合概率**::

    复合概率 = Laya 校准后概率 × 结构确定性系数

- **Laya 校准后概率**：取 ``confidence``（已压平虚高），不用原始 probabilities。
- **结构确定性系数**：由确定性代码核对的条件是否成立决定，范围 0.5~1.0。
  Laya 负责"倾向哪个方向"，结构位/ATR 负责"这个方向现在有没有条件做"。

两者的分工与项目既有设计一致（``LayaSettings`` 注释已写明）：
Laya 只提供概率，价格由确定性结构位 + ATR 计算。

三方向定义
----------
- **多**：Laya 方向偏多且结构有支撑可挂单
- **空**：Laya 方向偏空且结构有阻力可挂单
- **观望**：方向不明朗、信号无效、或多空条件均不成立时的默认分支
  （含噪音区/尺度冲突/突破失败等"不出手"信号）

概率与价格是**两件事**：三个分支都会给出完整三价位（entry/stop/target），
但概率高低反映的是"该不该做"；观望分支的价位仅作区间参考，不建议下单。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pa_agent.report.laya_pricing import PricePlan, plan_long, plan_short

#: 概率下限：只防负数/数值异常，**不做实质性钳制**。
#:
#: 早期版本用 0.10 兜底，结果多空两侧都被钳到同一值，排序完全失效
#: （报告显示"看起来一样可能"，实际拆分值差了近 2 倍）。因此这里
#: 只保留数值下限，真实高低由概率本身表达。
PROB_FLOOR = 0.0
#: 复合概率上限。刻意不给到 1.0——报告若出现"95% 胜率"必然是假象。
PROB_CEIL = 0.70

#: 低于此值视为"低概率"，报告里加标注提醒（**不改数值**）。
#: 零样本 Laya 方向题实测仅 0.14~0.17，扣完系数后低于 0.10 是常态，
#: 这本身就是"当前不该重仓 / 不该无条件下单"的有意义信号。
LOW_PROB_THRESHOLD = 0.10

#: 观望分支的固定概率基线。当多空均不成立时，观望就是"唯一合理选项"，
#: 给一个中位偏上的概率，避免出现"三个都很低"的观感。
WAIT_BASE_PROB = 0.55


@dataclass(frozen=True)
class Branch:
    """一个交易分支：方向 + 复合概率 + 价格计划 + 推导依据。"""

    key: str                    # long | short | wait
    label: str                  # 中文标签
    probability: float          # 复合概率 0~1
    plan: PricePlan
    #: 概率是怎么来的（人类可读），报告里必须展示，避免"黑箱概率"
    basis: str
    #: 是否属于低概率分支（< LOW_PROB_THRESHOLD）。仅用于报告标注，
    #: **不改变概率数值**——概率低本身就是有意义的信息。
    low_prob: bool = False
    #: 结构确定性系数明细
    factors: tuple[str, ...] = field(default_factory=tuple)
    #: 是否建议下单（观望分支为 False）
    actionable: bool = True


def _calibrated_conf(answers: dict[str, Any], qid: str) -> float | None:
    """取某题的**校准后**置信度；缺失返回 None。

    刻意不读 ``probabilities``——那是未校准的原始值，会虚高。
    """
    ans = answers.get(qid)
    if ans is None:
        return None
    conf = getattr(ans, "confidence", None)
    if conf is None and isinstance(ans, dict):
        conf = ans.get("confidence")
    try:
        return float(conf) if conf is not None else None
    except (TypeError, ValueError):
        return None


def _noul_value(answers: dict[str, Any], qid: str) -> float | None:
    """取noul 型答案的数值（信号有效/噪音区/尺度冲突）。"""
    ans = answers.get(qid)
    if ans is None:
        return None
    val = getattr(ans, "value", None)
    if val is None and isinstance(ans, dict):
        val = ans.get("value")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _direction_probs(answers: dict[str, Any]) -> tuple[float, float]:
    """返回 (看多概率, 看空概率)，按方向的**原始概率分布**拆分。

    为什么不用「方向」题的 ``confidence`` 直接当多空概率：
    ``confidence`` 是**胜出选项**的校准后置信度，实测仅 0.15，
    且它是bearish 或 bullish 单值，无法区分"多"和"空"哪个更可能。
    所以按 ``probabilities`` 里的 bullish / bearish 比例拆分，
    再整体乘一个校准压缩系数（把虚高拉回可信区间）。

    拆分后仍保留原始概率的**相对比例**（谁更可能），
    只压缩**绝对数值**——这是本模块与直接用原始概率的关键区别。
    """
    ans = answers.get("方向")
    if ans is None:
        return 0.30, 0.30
    probs = getattr(ans, "probabilities", None)
    if probs is None and isinstance(ans, dict):
        probs = ans.get("probabilities")
    conf = _calibrated_conf(answers, "方向")

    if not isinstance(probs, dict) or not probs:
        return (conf or 0.30), 0.0

    bull = float(probs.get("bullish", 0.0) or 0.0)
    bear = float(probs.get("bearish", 0.0) or 0.0)
    total = bull + bear
    if total <= 0:
        return (conf or 0.30), 0.0

    # 校准压缩系数：实测原始 0.5725→校准 0.1513，比值约 0.264。
    # 用它把拆分后的绝对值压回可信区间，同时保留 bull/bear 的相对比例。
    if conf is not None and conf > 0:
        #以胜出项为基准反推压缩系数（胜出项原始值→校准值）
        winner_raw = max(bull, bear)
        if winner_raw > 0:
            shrink = min(1.0, conf / winner_raw)
        else:
            shrink = 0.3
    else:
        shrink = 0.3

    return bull / total * shrink, bear / total * shrink


def _structure_factor(
    *,
    direction: str,
    has_structure: bool,
    rr: float | None,
    risk_in_atr: float | None,
    actionable: bool,
) -> tuple[float, list[str]]:
    """结构确定性系数：由可核对的条件决定，范围 0.5~1.0。

    刻意保守——条件不满足就压系数，而不是直接给0。
    因为"结构不完美"不等于"绝不能做"，只是需要人工复核。

    注意 RR 的处理：实测聚合数据下支撑可能贴现价极近（R 仅 0.35×ATR），
    而MM 投影目标很远，算出 RR=91 这种荒谬值。**这种"虚高 RR"必须扣分**，
    否则会把最差的结构误判成"盈亏比达标"。
    """
    reasons: list[str] = []
    factor = 1.0

    if not actionable:
        factor *= 0.5
        reasons.append("价格计划不可执行（结构位/ATR 缺失）→ ×0.5")
    elif not has_structure:
        # 纯 ATR 兜底，没有真实结构位支撑
        factor *= 0.75
        reasons.append("依赖 ATR 兜底、无真实结构位 → ×0.75")

    # 止损距离过近 → R 没有统计意义，盈亏比必然虚高
    if risk_in_atr is not None and risk_in_atr < 0.5:
        factor *= 0.6
        reasons.append(f"止损距离仅 {risk_in_atr:.2f}×ATR（<0.5），R 无统计意义 → ×0.6")

    if rr is not None:
        if rr < 1.0:
            factor *= 0.6
            reasons.append(f"盈亏比 {rr:.2f} < 1 → ×0.6")
        elif rr < 1.5:
            factor *= 0.8
            reasons.append(f"盈亏比 {rr:.2f} < 1.5 → ×0.8")
        elif rr > 10.0:
            # 虚高 RR：通常是止损贴太近或 MM 远投，不具参考价值
            factor *= 0.7
            reasons.append(f"盈亏比 {rr:.1f} > 10（虚高，多为远距离 MM 投影）→ ×0.7")
        else:
            reasons.append(f"盈亏比 {rr:.2f} 达标 → ×1.0")

    return max(0.5, min(1.0, factor)), reasons


def _signal_factor(answers: dict[str, Any]) -> tuple[float, list[str]]:
    """信号质量系数：看 Laya 的「信号有效 / 噪音区 / 尺度冲突」三题。

    这三题是 noul 型，值本身就是 0~1 的强度，无需再取概率。
    """
    reasons: list[str] = []
    factor = 1.0

    sig = _noul_value(answers, "信号有效")
    if sig is not None:
        if sig >= 0.65:
            reasons.append(f"信号有效 {sig:.2f}（高）→ ×1.0")
        elif sig >= 0.45:
            factor *= 0.85
            reasons.append(f"信号有效 {sig:.2f}（中）→ ×0.85")
        else:
            factor *= 0.6
            reasons.append(f"信号有效 {sig:.2f}（低）→ ×0.6")

    noise = _noul_value(answers, "噪音区")
    if noise is not None:
        if noise >= 0.65:
            factor *= 0.7
            reasons.append(f"噪音区 {noise:.2f}（偏高）→ ×0.7")
        elif noise >= 0.45:
            reasons.append(f"噪音区 {noise:.2f}（可接受）→ ×1.0")
        else:
            reasons.append(f"噪音区 {noise:.2f}（低）→ ×1.0")

    conflict = _noul_value(answers, "尺度冲突")
    if conflict is not None and conflict >= 0.65:
        factor *= 0.75
        reasons.append(f"尺度冲突 {conflict:.2f}（高）→ ×0.75")

    return max(0.4, min(1.0, factor)), reasons


def _wait_branch(
    *,
    close: float,
    atr: float | None,
    features: Any,
    answers: dict[str, Any],
    reason: str,
) -> Branch:
    """构造观望分支：给出区间参考价位，但明确不建议下单。

    观望的"三价位"取区间两端（上沿/下沿）作为伪 entry/stop，
    并用 zone 语义标注——**这不是交易计划，是观察区间**。
    """
    notes = ["观望分支：以下价位是观察区间参考，不是交易计划，不建议据此下单"]

    hi = None
    lo = None
    if getattr(features, "range_high", None) is not None:
        hi = float(features.range_high)
    if getattr(features, "range_low", None) is not None:
        lo = float(features.range_low)
    if hi is None or lo is None:
        # 连区间都没有 → 退回现价 ±ATR 作为观察带
        if atr and atr > 0 and close:
            hi = float(close) + float(atr)
            lo = float(close) - float(atr)
            notes.append("区间数据缺失，用现价 ±1×ATR 作为观察带")

    plan = PricePlan(
        direction="none",
        actionable=False,
        reason=reason,
        entry=hi,
        stop=lo,
        target=None,
        risk_per_unit=None,
        rr_ratio=None,
        notes=tuple(notes),
    )

    # 观望概率：取"不出手"的合理性，三个条件越明显越高
    factors: list[str] = []
    prob = WAIT_BASE_PROB
    sig = _noul_value(answers, "信号有效")
    if sig is not None and sig < 0.45:
        prob += 0.10
        factors.append(f"信号有效仅 {sig:.2f} → 观望更合理")
    noise = _noul_value(answers, "噪音区")
    if noise is not None and noise >= 0.65:
        prob += 0.10
        factors.append(f"噪音区 {noise:.2f} 偏高 → 观望更合理")
    conflict = _noul_value(answers, "尺度冲突")
    if conflict is not None and conflict >= 0.65:
        prob += 0.08
        factors.append(f"尺度冲突 {conflict:.2f} → 多空方向不明")
    bq_conf = _calibrated_conf(answers, "突破质量")
    if bq_conf is not None and bq_conf < 0.35:
        prob += 0.07
        factors.append(f"突破质量置信度 {bq_conf:.2f} 偏低 → 突破不可信")

    prob = min(PROB_CEIL, max(PROB_FLOOR, prob))
    basis = "观望 = 1 − (多/空置信度)，另加信号质量扣分项"
    return Branch(
        key="wait",
        label="观望",
        probability=prob,
        plan=plan,
        basis=basis,
        factors=tuple(factors) or ("无明显利多/利空信号，默认不出手",),
        actionable=False,
    )


def derive_branches(
    *,
    close: float,
    atr: float | None,
    features: Any,
    cfg: Any,
    answers: dict[str, Any],
    tick: float | None = None,
) -> list[Branch]:
    """从同一份数据推导三个交易分支，按复合概率降序返回。

    参数
    ----
    answers
        ``LayaAnswer`` 字典（键为中文问题名），取**校准后**置信度。
    tick
        品种最小变动价位。传入后价位按 tick 取整，与主报告一致；
        留 None 则保留 5 位小数。

    返回
    ----
    list[Branch]
        恰好 3 个，key 分别为 long / short / wait，按概率降序。
    """

    kwargs = dict(close=close, atr=atr, features=features, cfg=cfg, tick=tick)
    long_plan = plan_long(**kwargs)
    short_plan = plan_short(**kwargs)

    branches: list[Branch] = []

    # ── 信号质量系数（两方向共用）
    sig_factor, sig_reasons = _signal_factor(answers)

    # ── 方向概率：按 bull/bear 原始分布拆分 + 校准压缩
    dir_conf = _calibrated_conf(answers, "方向")
    long_model_prob, short_model_prob = _direction_probs(answers)

    def _risk_atr(plan: PricePlan) -> float | None:
        if plan.risk_per_unit and atr and atr > 0:
            return plan.risk_per_unit / atr
        return None

    # ── 多分支
    long_sf, long_sr = _structure_factor(
        direction="long",
        has_structure=bool(getattr(features, "supports", None)),
        rr=long_plan.rr_ratio,
        risk_in_atr=_risk_atr(long_plan),
        actionable=long_plan.actionable,
    )
    long_prob = min(PROB_CEIL, max(PROB_FLOOR, long_model_prob * sig_factor * long_sf))
    branches.append(
        Branch(
            key="long",
            label="做多",
            probability=long_prob,
            plan=long_plan,
            basis=(
                f"看多校准概率 {long_model_prob:.4f} × 信号质量 {sig_factor:.2f}"
                f" × 结构 {long_sf:.2f} = {long_prob:.2f}"
            ),
            factors=tuple(sig_reasons + long_sr),
            actionable=long_plan.actionable,
            low_prob=long_prob < LOW_PROB_THRESHOLD,
        )
    )

    # ── 空分支
    short_sf, short_sr = _structure_factor(
        direction="short",
        has_structure=bool(getattr(features, "resistances", None)),
        rr=short_plan.rr_ratio,
        risk_in_atr=_risk_atr(short_plan),
        actionable=short_plan.actionable,
    )
    short_prob = min(PROB_CEIL, max(PROB_FLOOR, short_model_prob * sig_factor * short_sf))
    branches.append(
        Branch(
            key="short",
            label="做空",
            probability=short_prob,
            plan=short_plan,
            basis=(
                f"看空校准概率 {short_model_prob:.4f} × 信号质量 {sig_factor:.2f}"
                f" × 结构 {short_sf:.2f} = {short_prob:.2f}"
            ),
            factors=tuple(sig_reasons + short_sr),
            actionable=short_plan.actionable,
            low_prob=short_prob < LOW_PROB_THRESHOLD,
        )
    )

    # ── 观望分支
    wait_reason = _decide_wait_reason(
        answers=answers,
        long_plan=long_plan,
        short_plan=short_plan,
        dir_conf=dir_conf,
        features=features,
    )
    branches.append(
        _wait_branch(
            close=close,
            atr=atr,
            features=features,
            answers=answers,
            reason=wait_reason,
        )
    )

    # 按概率降序；同概率时保持 多→空→观望 的固定顺序（不依赖 dict序）
    order = {"long": 0, "short": 1, "wait": 2}
    branches.sort(key=lambda b: (-b.probability, order[b.key]))
    return branches


def _decide_wait_reason(
    *,
    answers: dict[str, Any],
    long_plan: PricePlan,
    short_plan: PricePlan,
    dir_conf: float | None,
    features: Any,
) -> str:
    """给出观望的中文理由（列举触发观望的具体条件）。"""
    triggers: list[str] = []

    if dir_conf is not None and dir_conf < 0.35:
        triggers.append(f"方向置信度仅 {dir_conf:.0%}，低于 35% 门槛")

    if not long_plan.actionable and not short_plan.actionable:
        triggers.append("多空结构位均缺失，两个方向都无法给出有效价位")

    noise = _noul_value(answers, "噪音区")
    if noise is not None and noise >= 0.65:
        triggers.append(f"噪音区评分 {noise:.2f} 偏高，属铁丝网/无趋势区")

    conflict = _noul_value(answers, "尺度冲突")
    if conflict is not None and conflict >= 0.65:
        triggers.append("多尺度方向冲突，趋势在小周期与周期级不一致")

    bq_conf = _calibrated_conf(answers, "突破质量")
    if bq_conf is not None and bq_conf < 0.35:
        triggers.append(f"突破质量置信度 {bq_conf:.0%} 偏低，突破方向不可信")

    if not triggers:
        triggers.append("当前无明确利多或利空信号优势")
    return "；".join(triggers) + "，建议观望等待更明确机会"
