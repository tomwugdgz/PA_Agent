# -*- coding: utf-8 -*-
"""Laya 本地结构化报告：组装 + Markdown/HTML 渲染。

报告结构（纯本地，不经任何 LLM）
--------------------------------
1. 结论卡   — Laya 的方向/结构/信号有效性（含置信度与是否可信）
2. 价格计划 — plan_long / plan_short 的确定性输出
3. 判据明细 — 全部 Laya 原语答案（choice 概率分布 / noul P(true)）
4. 市场状态 — 喂给 Laya 的原文（可复核输入）
5. 页脚     — 模型、设备、耗时、免责

与两阶段 DeepSeek 分析**完全无关**：不读 records、不写 records、不共享提示词。
"""
from __future__ import annotations

import html
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pa_agent.ai.laya_schema import LayaPrediction
from pa_agent.report.laya_pricing import PricePlan

logger = logging.getLogger(__name__)

_DIRECTION_ZH = {"bullish": "偏多", "bearish": "偏空", "neutral": "中性"}
_ACT_ZH = {"long": "买入", "short": "卖出"}


@dataclass(frozen=True)
class LayaReport:
    """渲染前的完整报告数据。"""

    symbol: str
    timeframe: str
    generated_at: str
    prediction: LayaPrediction
    long_plan: PricePlan
    short_plan: PricePlan
    close: float | None
    atr: float | None
    supports: tuple[float, ...] = ()
    resistances: tuple[float, ...] = ()
    state_text: str = ""
    errors: tuple[str, ...] = field(default_factory=tuple)
    #: 三分支交易判断（多/空/观望 + 复合概率），由 report/laya_branches.py 生成。
    #: 放在最后并给默认值，保证旧调用方（只传前14 个字段）不受影响。
    branches: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """机器可读快照（写入标注数据集 / 供后续回测）。"""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "generated_at": self.generated_at,
            "close": self.close,
            "atr": self.atr,
            "supports": list(self.supports),
            "resistances": list(self.resistances),
            "prediction": {
                "answers": {
                    qid: {
                        "kind": a.kind,
                        "value": a.value,
                        "confidence": a.confidence,
                        "probabilities": a.probabilities,
                        "reliable": a.reliable,
                    }
                    for qid, a in self.prediction.answers.items()
                },
                "usage": self.prediction.usage,
                "latency_ms": self.prediction.latency_ms,
                "device": self.prediction.device,
                "calibrated": self.prediction.calibrated,
            },
            "long_plan": _plan_dict(self.long_plan),
            "short_plan": _plan_dict(self.short_plan),
            "branches": [_branch_dict(b) for b in self.branches],
            "errors": list(self.errors),
        }


def _branch_dict(b: Any) -> dict[str, Any]:
    """把一个 Branch 序列化成机器可读结构。

    刻意用鸭子类型（``getattr``）而非 isinstance——``laya_branches`` 里的
    ``Branch`` 是 frozen dataclass，但分支推导是可选功能，报告渲染时
    拿到 dict 形态（历史 JSON 回读）也不该崩。
    """
    if isinstance(b, dict):
        return dict(b)
    plan = getattr(b, "plan", None)
    return {
        "key": getattr(b, "key", ""),
        "label": getattr(b, "label", ""),
        "probability": round(float(getattr(b, "probability", 0.0)), 4),
        "actionable": bool(getattr(b, "actionable", False)),
        "low_prob": bool(getattr(b, "low_prob", False)),
        "basis": getattr(b, "basis", ""),
        "factors": list(getattr(b, "factors", ()) or ()),
        "plan": _plan_dict(plan) if plan is not None else None,
    }


def _pred_get(pred, key, default=None):
    """兼容 dict / dataclass / SimpleNamespace 的 get 操作。"""
    if hasattr(pred, "get") and not isinstance(pred, dict):
        # dataclass/SimpleNamespace：从 answers 属性里取
        answers = getattr(pred, "answers", {})
        if isinstance(answers, dict):
            return answers.get(key, default)
        return default
    elif isinstance(pred, dict):
        return pred.get(key, default)
    return default

def _plan_dict(p: PricePlan) -> dict[str, Any]:
    return {
        "direction": p.direction,
        "actionable": p.actionable,
        "reason": p.reason,
        "entry": p.entry,
        "stop": p.stop,
        "target": p.target,
        "risk_per_unit": p.risk_per_unit,
        "rr_ratio": p.rr_ratio,
        "fallbacks": {
            "entry": p.entry_fallback,
            "stop": p.stop_fallback,
            "target": p.target_fallback,
        },
        "notes": list(p.notes),
    }


# ── Markdown 渲染 ────────────────────────────────────────────────────────────


def _branches_md(r: LayaReport, L: list[str]) -> None:
    """三分支判断的 Markdown 段落（与 HTML 卡片同源同数据）。"""
    branches = list(getattr(r, "branches", ()) or ())
    if not branches:
        return

    L.append("## 二、三分支判断（同一份数据推导）")
    L.append("")
    L.append(
        "> 概率 = Laya **校准后**概率 × 信号质量系数 × 结构确定性系数。"
        "刻意不使用 Laya 原始 `probabilities`——那是未校准值，"
        "方向题实测虚高约 1.94 倍（原始 0.57 → 校准后 0.15），"
        "直接排序会给出接近随机的假高概率。"
    )
    L.append("")

    for rank, b in enumerate(branches, 1):
        if isinstance(b, dict):
            continue
        key = str(getattr(b, "key", "") or "wait")
        label = getattr(b, "label", key)
        prob = float(getattr(b, "probability", 0.0) or 0.0)
        plan = getattr(b, "plan", None)

        L.append(f"### #{rank} {label}　复合概率 {prob:.0%}")
        L.append("")
        if bool(getattr(b, "low_prob", False)) and key != "wait":
            L.append("> ⚠️ **低概率分支（<10%）**：建议观望或极小仓位。")
            L.append("")
        L.append(f"- 概率构成：{getattr(b, 'basis', '')}")
        for f in getattr(b, "factors", ()) or ():
            L.append(f"  - {f}")

        if plan is not None:
            if not getattr(plan, "actionable", False) and key == "wait":
                L.append(f"- 观察区间：{plan.stop} ～ {plan.entry}（**非交易计划，不建议下单**）")
                L.append(f"- 观望理由：{plan.reason}")
            elif not getattr(plan, "actionable", False):
                L.append(f"- 不出价：{plan.reason}")
            else:
                rr = f"，盈亏比 ≈ {plan.rr_ratio}" if plan.rr_ratio else ""
                L.append(
                    f"- 挂单价 **{plan.entry}**　止损 **{plan.stop}**"
                    f"　目标 **{plan.target}**{rr}"
                )
            for n in getattr(plan, "notes", ()) or ():
                L.append(f"  - {n}")
        L.append("")


def render_markdown(r: LayaReport) -> str:
    """人类可读的中文报告（纯文本 Markdown，可直接粘贴/存档）。"""
    L: list[str] = []
    pred = r.prediction
    struct = _pred_get(pred, "周期结构")
    dire = _pred_get(pred, "方向")
    sig = _pred_get(pred, "信号有效")

    L.append(f"# Laya 市场分析报告 · {r.symbol} {r.timeframe}")
    L.append("")
    L.append(f"- 生成时间：{r.generated_at}")
    L.append(f"- 推理设备：{pred.device or '未知'}（加载 {pred.load_ms / 1000:.1f}s，推理 {pred.latency_ms:.0f}ms）")
    L.append(f"- 现价：{r.close if r.close is not None else '—'}　ATR14：{r.atr if r.atr is not None else '—'}")
    cal = "已启用" if getattr(pred, "calibrated", False) else \
        "未启用（零样本置信度天然偏低，建议先积累标注再校准）"
    L.append(f"- 置信度校准：{cal}")
    L.append("")

    # ── 1. 结论卡
    L.append("## 一、结论")
    L.append("")
    if dire:
        conf_pct = f"{dire.confidence * 100:.0f}%"
        mark = "" if dire.reliable else "（⚠️ 置信度低于门槛，仅供参考）"
        L.append(f"- **方向倾向：{_DIRECTION_ZH.get(str(dire.value), dire.value)}**（置信度 {conf_pct}）{mark}")
    if struct:
        conf_pct = f"{struct.confidence * 100:.0f}%"
        mark = "" if struct.reliable else "（⚠️ 低置信）"
        L.append(f"- 周期结构：{struct.value}（置信度 {conf_pct}）{mark}")
    if sig:
        mark = "" if sig.reliable else "（⚠️ 低置信）"
        L.append(f"- 信号有效性：P(存在可靠信号) = {float(sig.value):.2f}{mark}")
    low_conf = [a for a in pred.answers.values() if not a.reliable]
    if low_conf:
        L.append(f"- ⚠️ {len(low_conf)} 项答案低于置信度门槛，相关价格建议已标注为不可靠。")
    L.append("")

    # ── 2. 三分支判断
    _branches_md(r, L)

    # ── 3. 价格计划
    L.append("## 三、价格计划（确定性计算：结构位优先 + ATR 兜底）")
    L.append("")
    for plan in (r.long_plan, r.short_plan):
        act = _ACT_ZH.get(plan.direction, plan.direction)
        L.append(f"### {act}计划（{plan.direction}）")
        L.append("")
        if not plan.actionable:
            L.append(f"- ❌ 不出价：{plan.reason}")
        else:
            L.append(f"- 挂单价：**{plan.entry}**")
            L.append(f"- 止损：**{plan.stop}**（1 单位风险 R = {plan.risk_per_unit}）")
            L.append(f"- 目标：**{plan.target}**" + (f"（盈亏比 ≈ {plan.rr_ratio}）" if plan.rr_ratio else ""))
            for n in plan.notes:
                L.append(f"- 依据：{n}")
        L.append("")
    if r.supports or r.resistances:
        L.append("**原始结构位**")
        L.append("")
        if r.supports:
            L.append(f"- 支撑（近→远）：{'、'.join(str(p) for p in r.supports[:3])}")
        if r.resistances:
            L.append(f"- 阻力（近→远）：{'、'.join(str(p) for p in r.resistances[:3])}")
        L.append("")

    # ── 3. 判据明细
    L.append("## 四、判据明细（Laya 原语输出）")
    L.append("")
    L.append("| 问题 | 类型 | 答案 | 置信度 | 可信 |")
    L.append("|---|---|---|---|---|")
    for qid, a in pred.answers.items():
        if a.kind == "choice":
            shown = _DIRECTION_ZH.get(str(a.value), str(a.value))
        elif a.kind == "noul":
            shown = f"P(true) = {float(a.value):.3f}"
        else:
            shown = str(a.value)
        L.append(
            f"| {qid} | {a.kind} | {shown} | {a.confidence * 100:.0f}% | "
            f"{'✅' if a.reliable else '⚠️'} |"
        )
    L.append("")

    # choice 概率分布展开（方向 + 周期结构）
    for qid in ("方向", "周期结构", "突破质量"):
        a = _pred_get(pred, qid)
        if a and a.probabilities:
            L.append(f"**{qid} 概率分布**")
            L.append("")
            for k, v in sorted(a.probabilities.items(), key=lambda kv: -kv[1]):
                L.append(f"- {_DIRECTION_ZH.get(k, k)}：{v * 100:.1f}%")
            L.append("")

    # ── 4. 市场状态
    L.append("## 五、喂给模型的市场状态（可复核）")
    L.append("")
    L.append("```text")
    L.append(r.state_text)
    L.append("```")
    L.append("")

    # ── 5. 页脚
    L.append("---")
    L.append("")
    L.append(
        "> **免责**：本报告由本地确定性代码 + Laya 判别模型（System 1）生成，"
        "非投资建议。Laya 未微调时置信度不可靠，仅作决策辅助；"
        "所有价格均由结构位/ATR 公式计算，可复现、可回测。"
    )
    if r.errors:
        L.append("")
        L.append("**生成过程异常**：")
        for e in r.errors:
            L.append(f"- {e}")
    return "\n".join(L) + "\n"


# ── HTML 渲染 ────────────────────────────────────────────────────────────────

_HTML_TMPL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Laya 报告 · {symbol} {timeframe}</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #F5F5F7; color: #1D1D1F;
         font: 15px/1.7 -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; }}
  .wrap {{ max-width: 860px; margin: 0 auto; padding: 40px 24px 80px; }}
  h1 {{ font-size: 26px; margin: 0 0 6px; letter-spacing: .5px; }}
  h2 {{ font-size: 18px; margin: 34px 0 12px; padding-bottom: 8px;
        border-bottom: 1px solid #E5E5EA; }}
  .meta {{ color: #6E6E73; font-size: 13px; margin-bottom: 24px; }}
  .card {{ background: #fff; border: 1px solid #E5E5EA; border-radius: 12px;
           padding: 20px 24px; margin: 14px 0; }}
  .big {{ font-size: 30px; font-weight: 700; margin: 4px 0; }}
  .long .big {{ color: #D62728; }}   /* 涨=红，跌=绿：A股口径 */
  .short .big {{ color: #1A9850; }}
  .muted {{ color: #6E6E73; font-size: 13px; }}
  .warn {{ color: #B45309; background: #FEF3C7; border: 1px solid #FDE68A;
           border-radius: 8px; padding: 10px 14px; font-size: 13px; margin: 12px 0; }}
  .price {{ font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff;
           border: 1px solid #E5E5EA; border-radius: 12px; overflow: hidden; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #F0F0F2;
            font-size: 14px; }}
  th {{ background: #FAFAFC; color: #6E6E73; font-weight: 600; font-size: 13px; }}
  tr:last-child td {{ border-bottom: none; }}
  pre {{ background: #1D1D1F; color: #F5F5F7; border-radius: 10px;
         padding: 16px 18px; font: 12px/1.6 Consolas, monospace;
         overflow-x: auto; white-space: pre-wrap; }}
  .bar {{ height: 10px; background: #F0F0F2; border-radius: 5px; overflow: hidden;
          margin: 4px 0 10px; }}
  .bar i {{ display: block; height: 100%; background: #FF5C1A; }}
  .foot {{ margin-top: 36px; color: #6E6E73; font-size: 12px;
           border-top: 1px solid #E5E5EA; padding-top: 16px; }}
  .na {{ color: #8E8E93; }}
  /* ── 三分支卡片 ── */
  .branch {{ background: #fff; border: 1px solid #E5E5EA; border-radius: 12px;
            padding: 18px 22px; margin: 12px 0; }}
  .branch.long {{ border-left: 4px solid #D62728; }}
  .branch.short {{ border-left: 4px solid #1A9850; }}
  .branch.wait {{ border-left: 4px solid #8E8E93; }}
  .branch-hd {{ display: flex; align-items: baseline; gap: 10px;
               flex-wrap: wrap; margin-bottom: 6px; }}
  .branch-rank {{ font-size: 12px; color: #fff; background: #1D1D1F;
                 border-radius: 999px; padding: 2px 9px; font-weight: 600; }}
  .branch-rank.long {{ background: #D62728; }}
  .branch-rank.short {{ background: #1A9850; }}
  .branch-rank.wait {{ background: #8E8E93; }}
  .branch-name {{ font-size: 17px; font-weight: 700; }}
  .branch-prob {{ margin-left: auto; font-size: 20px; font-weight: 700;
                  font-variant-numeric: tabular-nums; }}
  .branch-prob.long {{ color: #D62728; }}
  .branch-prob.short {{ color: #1A9850; }}
  .branch-prob.wait {{ color: #6E6E73; }}
  .pbar {{ height: 8px; background: #F0F0F2; border-radius: 4px;
           overflow: hidden; margin: 8px 0 12px; }}
  .pbar i {{ display: block; height: 100%; }}
  .pbar.long i {{ background: #D62728; }}
  .pbar.short i {{ background: #1A9850; }}
  .pbar.wait i {{ background: #8E8E93; }}
  .basis {{ background: #FAFAFC; border: 1px solid #F0F0F2; border-radius: 8px;
           padding: 9px 12px; font-size: 12px; color: #6E6E73; margin: 8px 0; }}
  .basis b {{ color: #1D1D1F; font-weight: 600; }}
  .lv {{ display: flex; gap: 18px; flex-wrap: wrap; margin: 8px 0;
         font-variant-numeric: tabular-nums; }}
  .lv > div {{ font-size: 13px; }}
  .lv span {{ color: #6E6E73; font-size: 12px; }}
  .lv b {{ font-size: 15px; }}
  .tag-no {{ display: inline-block; background: #FEF3C7; border: 1px solid #FDE68A;
            color: #B45309; border-radius: 6px; padding: 1px 8px; font-size: 12px; }}
</style>
</head>
<body><div class="wrap">
<h1>Laya 市场分析报告</h1>
<div class="meta">{symbol} · {timeframe} · {generated_at} · 设备 {device} · 推理 {latency_ms:.0f}ms</div>

{errors_html}

{mt5_action_html}

{trust_html}

{memory_html}

<h2>结论</h2>
<div class="card">
  <div class="big">{direction_zh}</div>
  <div class="muted">方向置信度 {direction_conf}</div>
  {struct_line}
  {signal_line}
</div>
{low_conf_html}

{branches_html}

<h2>价格计划 <span class="muted">（结构位优先 + ATR 兜底，确定性计算）</span></h2>
{plans_html}

<h2>判据明细</h2>
<table>
<tr><th>问题</th><th>类型</th><th>答案</th><th>置信度</th><th>可信</th></tr>
{detail_rows}
</table>

{prob_html}

<h2>喂给模型的市场状态</h2>
<pre>{state_text}</pre>

<div class="foot">
现价 {close} · ATR14 {atr}<br>
支撑：{supports_txt}　阻力：{resistances_txt}<br><br>
免责：本报告由本地确定性代码 + Laya 判别模型（System 1）生成，非投资建议。
Laya 未微调时置信度不可靠，仅作决策辅助；所有价格均由结构位/ATR 公式计算，可复现、可回测。
</div>
</div></body></html>
"""


def _branches_html(r: LayaReport) -> str:
    """三分支交易判断卡片：多/空/观望 + 复合概率 + 完整三价位。

    概率是「Laya 校准后置信度 × 信号质量 × 结构确定性」的复合值，
    并在卡片里把算法写出来——不给黑箱数字。
    """
    branches = list(getattr(r, "branches", ()) or ())
    if not branches:
        return ""

    rows: list[str] = [
        '<h2>三分支判断（同一份数据推导）</h2>',
        '<div class="warn">概率 = Laya <b>校准后</b>置信度 × 信号质量系数 × 结构确定性系数。'
        '刻意不使用 Laya 原始 probabilities——那是未校准值，方向题实测虚高约 1.94 倍'
        '（原始 0.57 → 校准后 0.15），直接排序会给出接近随机的假高概率。</div>',
    ]

    for rank, b in enumerate(branches, 1):
        if isinstance(b, dict):
            continue
        key = str(getattr(b, "key", "") or "wait")
        label = html.escape(str(getattr(b, "label", key)))
        prob = float(getattr(b, "probability", 0.0) or 0.0)
        basis = html.escape(str(getattr(b, "basis", "")))
        factors = list(getattr(b, "factors", ()) or ())
        plan = getattr(b, "plan", None)
        actionable = bool(getattr(b, "actionable", False))
        low_prob = bool(getattr(b, "low_prob", False)) and key != "wait"

        pct = max(0.0, min(1.0, prob)) * 100.0
        factors_html = (
            "<div class='basis'><b>扣分明细</b><br>"
            + "<br>".join(html.escape(str(f)) for f in factors)
            + "</div>"
            if factors
            else ""
        )

        if plan is None:
            lv_html = "<div class='muted'>无价格计划</div>"
        elif not actionable and key == "wait":
            # 观望：价位是观察区间，不是交易计划
            lv_html = (
                f"<div class='lv'>"
                f"<div><span>区间上沿 </span><b>{plan.entry}</b></div>"
                f"<div><span>区间下沿 </span><b>{plan.stop}</b></div>"
                f"</div>"
                f"<div class='muted'>{html.escape(plan.reason)}</div>"
                f"<div><span class='tag-no'>不建议下单</span></div>"
            )
        elif not actionable:
            lv_html = (
                f"<div class='price na'>不出价</div>"
                f"<div class='muted'>{html.escape(plan.reason)}</div>"
            )
        else:
            rr_txt = f"<span class='muted'>（R = {plan.risk_per_unit}）</span>" if plan.risk_per_unit else ""
            rr2 = f"<span class='muted'>（盈亏比 ≈ {plan.rr_ratio}）</span>" if plan.rr_ratio else ""
            notes = "".join(
                f"<div class='muted'>{html.escape(n)}</div>" for n in (plan.notes or ())
            )
            lv_html = (
                f"<div class='lv'>"
                f"<div><span>挂单价 </span><b class='price'>{plan.entry}</b></div>"
                f"<div><span>止损 </span><b class='price'>{plan.stop}</b>{rr_txt}</div>"
                f"<div><span>目标 </span><b class='price'>{plan.target}</b>{rr2}</div>"
                f"</div>" + notes
            )

        rows.append(
            f'<div class="branch {key}">'
            f'<div class="branch-hd">'
            f'<span class="branch-rank {key}">#{rank}</span>'
            f'<span class="branch-name">{label}</span>'
            f'<span class="branch-prob {key}">{prob:.0%}</span>'
            f"</div>"
            f'<div class="pbar {key}"><i style="width:{pct:.1f}%"></i></div>'
            f'<div class="basis"><b>概率构成</b> · {basis}</div>'
            f"{factors_html}"
            f"{'<div><span class=\'tag-no\'>低概率分支（<10%），建议观望或极小仓位</span></div>' if low_prob else ''}"
            f"{lv_html}"
            f"</div>"
        )

    return "\n".join(rows)


def _plan_html(p: PricePlan) -> str:
    act = _ACT_ZH.get(p.direction, p.direction)
    cls = "long" if p.direction == "long" else "short"
    if not p.actionable:
        return (
            f'<div class="card {cls}"><div class="muted">{html.escape(act)}计划</div>'
            f'<div class="price na">不出价</div>'
            f'<div class="muted">{html.escape(p.reason)}</div></div>'
        )
    rows = [
        f"<div>挂单价 <span class='price'>{p.entry}</span></div>",
        f"<div>止损 <span class='price'>{p.stop}</span>"
        f"<span class='muted'>（R = {p.risk_per_unit}）</span></div>",
        f"<div>目标 <span class='price'>{p.target}</span>"
        + (f"<span class='muted'>（盈亏比 ≈ {p.rr_ratio}）</span>" if p.rr_ratio else "")
        + "</div>",
    ]
    notes = "".join(f"<div class='muted'>{html.escape(n)}</div>" for n in p.notes)
    return (
        f'<div class="card {cls}"><div class="muted">{html.escape(act)}计划</div>'
        + "".join(rows) + notes + "</div>"
    )


def _mt5_action_card(r: LayaReport) -> str:
    """根据报告方向生成 MT5 操作指引卡片。"""
    pred = r.prediction
    # 兼容 dict 和 dataclass/SimpleNamespace
    if hasattr(pred, "get"):
        dire = _pred_get(pred, "方向")
    else:
        dire = getattr(pred, "answers", {}).get("方向") if isinstance(getattr(pred, "answers", None), dict) else None
    if not dire or not isinstance(getattr(dire, "value", None), str):
        return ""

    direction = str(dire.value).lower()
    # 选可执行的那个计划
    plan = None
    if direction == "long" and r.long_plan.actionable:
        plan = r.long_plan
    elif direction == "short" and r.short_plan.actionable:
        plan = r.short_plan
    else:
        # 两边都试试
        for p in (r.long_plan, r.short_plan):
            if p.actionable:
                plan = p
                direction = p.direction
                break
    if not plan:
        return ""

    side_en = "Buy" if direction == "long" else "Sell"
    kind_map = {"limit": "Limit", "stop": "Stop"}
    # 判断类型：如果 entry 接近支撑/阻力位，默认限价；否则突破
    order_type = "Limit"  # 默认
    entry_str = f"{plan.entry}" if plan.entry else "市价"

    button_name = f"{side_en} {order_type}" if order_type != "Market" else side_en

    card_class = "long" if direction == "long" else "short"
    arrow = "↑" if direction == "long" else "↓"
    zh_side = "买入" if direction == "long" else "卖出"

    steps = [f"<b>在 MT5 中点「{button_name}」</b>"]
    if order_type != "Market" and plan.entry:
        steps.append(f"Price（价格）填 <b>{plan.entry}</b>")
    steps.append(f"Volume（手数）建议 <b>1.00</b>（可在面板修改）")
    if plan.stop:
        steps.append(f"Stop Loss（止损）填 <b>{plan.stop}</b>")
    if plan.target:
        steps.append(f"Take Profit（止盈）填 <b>{plan.target}</b>")
    steps.append("最后点「下订单」或按 Enter")

    html_parts = [
        f"<div class='card {card_class}' style='border-left: 4px solid {'#D62728' if direction == 'long' else '#1A9850'};'>",
        f"<div style='font-size: 18px; font-weight: 700; margin-bottom: 8px;'>👉 MT5 操作指引：{zh_side}{arrow}</div>",
        "<ol style='margin: 0; padding-left: 20px; line-height: 1.8;'>",
    ]
    for s in steps:
        html_parts.append(f"<li>{s}</li>")
    html_parts.append("</ol></div>")
    return "".join(html_parts)


def _memory_card(r: LayaReport) -> str:
    """品种记忆卡片：本次均线快照 + 该品种历史命中率。

    这是"记忆库"在报告里的唯一入口，目的只有两个：
      1. 让人能确认**这份分析用的是哪一批数据**（均线快照随行情变）
      2. 让人知道**这个品种的历史预测准不准**（命中率高才值得参考）
    """
    try:
        from pa_agent.memory.settle import accuracy_hint
        from pa_agent.memory.symbol_memory import load_profile, symbol_dir
    except Exception as exc:  # noqa: BLE001
        logger.debug("记忆库卡片不可用: %s", exc)
        return ""

    # ── 本次均线快照（从state 之外拿不到，直接从画像读最近一次会话）──
    mas_line = ""
    try:
        sess_path = symbol_dir(r.symbol, r.timeframe) / "sessions.jsonl"
        if sess_path.is_file():
            lines = [ln for ln in sess_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                last = json.loads(lines[-1])
                mas = last.get("mas") or {}
                if mas:
                    parts = []
                    for p in (5, 10, 20, 60):
                        v = mas.get(str(p))
                        if v is not None:
                            parts.append(f"MA{p} {float(v):.5f}")
                    if parts:
                        pat = last.get("ma_pattern", "unknown")
                        pat_zh = {"bullish": "多头排列", "bearish": "空头排列",
                                  "mixed": "交织"}.get(pat, pat)
                        mas_line = (
                            f'<div class="muted">均线快照（{pat_zh}）：'
                            + "　".join(parts)
                            + "</div>"
                        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("均线快照读取失败: %s", exc)

    # ── 历史命中率 ──
    prof = load_profile(r.symbol, r.timeframe)
    if not prof.n_sessions:
        return ""
    hint = accuracy_hint(r.symbol, r.timeframe)
    acc = prof.direction_accuracy
    if acc is None:
        acc_html = "<span class='na'>暂无可统计的方向命中</span>"
    elif acc >= 0.6:
        acc_html = f"<b style='color:#1A9850'>{acc:.0%}</b>"
    elif acc >= 0.45:
        acc_html = f"<b style='color:#B45309'>{acc:.0%}</b>"
    else:
        acc_html = f"<b style='color:#D62728'>{acc:.0%}</b>"

    extra = ""
    if prof.atr_avg:
        extra += f"<div class='muted'>历史平均 ATR {prof.atr_avg}</div>"
    if prof.price_min is not None and prof.price_max is not None:
        extra += f"<div class='muted'>历史价格区间 {prof.price_min} ~ {prof.price_max}</div>"
    if prof.common_supports:
        extra += (
            "<div class='muted'>历史常见支撑 "
            + "、".join(f"{v}" for v in prof.common_supports[:3])
            + "</div>"
        )
    if prof.common_resistances:
        extra += (
            "<div class='muted'>历史常见阻力 "
            + "、".join(f"{v}" for v in prof.common_resistances[:3])
            + "</div>"
        )

    return (
        f'<div class="card">'
        f'<div class="muted">品种记忆库 · {html.escape(r.symbol)} {html.escape(r.timeframe)}</div>'
        f'<div class="big">历史方向命中率 {acc_html}</div>'
        f'<div class="muted">{html.escape(hint)}</div>'
        f'{mas_line}{extra}'
        f'<div class="muted" style="margin-top:6px">'
        f"已积累 {prof.n_sessions} 次分析 / {prof.n_settled} 次有效结算，"
        f"记录目录：experience/memory/{html.escape(r.symbol)}/{html.escape(r.timeframe)}/"
        f"</div>"
        f"</div>"
    )


def _confidence_trust_card(r: LayaReport) -> str:
    """置信度可信度卡片——明确告诉用户「这份结论该信到什么程度」。

    Laya 零样本的置信度普遍在 0.14~0.17（作者原话 near chance），直接照着
    下单等于抛硬币。本卡片做三件事：
      1. 显示校准状态（有没有挂载 calibration.json）
      2. 给置信度分级（够不够格据此下单）
      3. 明确写出「不可信时该怎么办」——不是只报一个数字
    """
    pred = r.prediction
    answers = getattr(pred, "answers", {}) or {}
    if not isinstance(answers, dict):
        return ""
    # 方向题是唯一直接驱动买卖的题，优先看它
    dire = answers.get("方向")
    conf = float(getattr(dire, "confidence", 0.0) or 0.0) if dire else 0.0
    reliable = bool(getattr(dire, "reliable", False)) if dire else False
    calibrated = bool(getattr(pred, "calibrated", False))

    # 分级：门槛与 settings.laya.min_confidence 保持一致（0.35）
    if conf >= 0.60:
        level, color, advice = "高", "#1A9850", "可作为下单依据，但仍建议叠加你自己的判断"
    elif conf >= 0.35:
        level, color, advice = "中", "#B8860B", "仅作参考，建议等信号更明确再动手"
    elif conf >= 0.20:
        level, color, advice = "低", "#D2691E", "不建议据此下单，先看结构位和回测结果"
    else:
        level, color, advice = "极低（接近随机）", "#D62728", "不要据此下单——这是模型的猜测，不是判断"

    cal_line = (
        "<b>已启用</b>" if calibrated else
        "<b>未启用</b>——零样本置信度天然偏低，"
        "建议先跑 <code>tools/weak_label.py</code> 积累样本，"
        "再跑 <code>tools/calibrate_laya.py</code>"
    )
    bits = [f"<div class='card' style='border-left: 4px solid {color};'>"]
    bits.append("<div style='font-size: 16px; font-weight: 700; margin-bottom: 6px;'>"
                "📊 置信度可信度</div>")
    bits.append(
        f"<div style='line-height: 1.9;'>"
        f"方向判断置信度：<b style='color: {color}; font-size: 15px;'>{conf:.0%}</b>"
        f"（等级：{level}）<br>"
        f"置信度校准：{cal_line}"
        f"<br>建议：{html.escape(advice)}</div>"
    )
    if not reliable:
        bits.append(
            "<div style='margin-top: 6px; color: #D62728;'>"
            "⚠️ 该结论未跨过可靠性门槛，价格计划仅供参考，不构成交易建议。</div>"
        )
    bits.append("</div>")
    return "".join(bits)


def render_html(r: LayaReport) -> str:
    """HTML 报告。极简未来主义：#F5F5F7 底 / #1D1D1F 字 / #FF5C1A 强调。"""
    pred = r.prediction
    # 兼容 dict / dataclass / SimpleNamespace
    if hasattr(pred, "get"):
        dire = _pred_get(pred, "方向")
        struct = _pred_get(pred, "周期结构")
        sig = _pred_get(pred, "信号有效")
    else:
        answers = getattr(pred, "answers", {})
        dire = answers.get("方向") if isinstance(answers, dict) else None
        struct = answers.get("周期结构") if isinstance(answers, dict) else None
        sig = answers.get("信号有效") if isinstance(answers, dict) else None

    direction_zh = _DIRECTION_ZH.get(str(dire.value if dire else ""), "—")
    detail_rows = []
    for qid, a in pred.answers.items():
        if a.kind == "choice":
            shown = _DIRECTION_ZH.get(str(a.value), str(a.value))
        elif a.kind == "noul":
            shown = f"P(true) = {float(a.value):.3f}"
        else:
            shown = str(a.value)
        detail_rows.append(
            f"<tr><td>{html.escape(qid)}</td><td>{a.kind}</td>"
            f"<td>{html.escape(shown)}</td><td>{a.confidence * 100:.0f}%</td>"
            f"<td>{'✅' if a.reliable else '⚠️'}</td></tr>"
        )

    prob_blocks = []
    for qid in ("方向", "周期结构", "突破质量"):
        a = _pred_get(pred, qid)
        if not (a and a.probabilities):
            continue
        items = []
        for k, v in sorted(a.probabilities.items(), key=lambda kv: -kv[1]):
            items.append(
                f"<div class='muted'>{html.escape(_DIRECTION_ZH.get(k, k))} {v * 100:.1f}%</div>"
                f"<div class='bar'><i style='width:{v * 100:.1f}%'></i></div>"
            )
        prob_blocks.append(
            f"<h2>{html.escape(qid)} 概率分布</h2><div class='card'>{''.join(items)}</div>"
        )

    low_conf = [a for a in pred.answers.values() if not a.reliable]
    low_conf_html = ""
    if low_conf:
        low_conf_html = (
            "<div class='warn'>⚠️ "
            f"{len(low_conf)} 项答案低于置信度门槛（"
            + "、".join(html.escape(a.qid) for a in low_conf)
            + "），相关价格建议仅供参考。</div>"
        )

    errors_html = ""
    if r.errors:
        errors_html = "<div class='warn'>生成异常：" + html.escape("；".join(r.errors)) + "</div>"

    # ── MT5 操作指引卡片（给新手看） ────────────────────────────────
    mt5_action_html = _mt5_action_card(r)
    # ── 置信度可信度卡片（明确该不该信） ────────────────────────────
    trust_html = _confidence_trust_card(r)
    # ── 品种记忆库卡片（历史命中率 + 本次均线快照） ────────────────
    memory_html = _memory_card(r)
    # ── 三分支交易判断（多/空/观望 + 复合概率） ─────────────────────
    branches_html = _branches_html(r)

    fmt = lambda v: "—" if v is None else str(v)  # noqa: E731
    return _HTML_TMPL.format(
        symbol=html.escape(r.symbol),
        timeframe=html.escape(r.timeframe),
        generated_at=html.escape(r.generated_at),
        device=html.escape(pred.device or "?"),
        latency_ms=pred.latency_ms,
        errors_html=errors_html,
        mt5_action_html=mt5_action_html,
        trust_html=trust_html,
        memory_html=memory_html,
        branches_html=branches_html,
        direction_zh=direction_zh,
        direction_conf=(f"{dire.confidence * 100:.0f}%" if dire else "—"),
        struct_line=(
            f"<div class='muted'>周期结构：{html.escape(str(struct.value))}"
            f"（{struct.confidence * 100:.0f}%）</div>" if struct else ""
        ),
        signal_line=(
            f"<div class='muted'>信号有效性 P = {float(sig.value):.2f}</div>" if sig else ""
        ),
        low_conf_html=low_conf_html,
        plans_html=_plan_html(r.long_plan) + _plan_html(r.short_plan),
        detail_rows="".join(detail_rows),
        prob_html="".join(prob_blocks),
        state_text=html.escape(r.state_text),
        close=fmt(r.close),
        atr=fmt(r.atr),
        supports_txt=html.escape("、".join(str(p) for p in r.supports[:3]) or "—"),
        resistances_txt=html.escape("、".join(str(p) for p in r.resistances[:3]) or "—"),
    )


def save_pair(md: str, html_text: str, base_path: Any) -> tuple[Any, Any]:
    """把 MD 与 HTML 写到 base_path 同名不同扩展名；返回两个实际路径。"""
    md_path = base_path.with_suffix(".md")
    html_path = base_path.with_suffix(".html")
    md_path.write_text(md, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")
    return md_path, html_path


def timestamp_slug() -> str:
    """报告文件名的时间片段：YYYYMMDD_HHMMSS。"""
    return time.strftime("%Y%m%d_%H%M%S")


def json_dumps(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
