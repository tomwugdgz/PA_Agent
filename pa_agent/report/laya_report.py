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
import time
from dataclasses import dataclass, field
from typing import Any

from pa_agent.ai.laya_schema import LayaPrediction
from pa_agent.report.laya_pricing import PricePlan

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
            },
            "long_plan": _plan_dict(self.long_plan),
            "short_plan": _plan_dict(self.short_plan),
            "errors": list(self.errors),
        }


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


def render_markdown(r: LayaReport) -> str:
    """人类可读的中文报告（纯文本 Markdown，可直接粘贴/存档）。"""
    L: list[str] = []
    pred = r.prediction
    struct = pred.get("周期结构")
    dire = pred.get("方向")
    sig = pred.get("信号有效")

    L.append(f"# Laya 市场分析报告 · {r.symbol} {r.timeframe}")
    L.append("")
    L.append(f"- 生成时间：{r.generated_at}")
    L.append(f"- 推理设备：{pred.device or '未知'}（加载 {pred.load_ms / 1000:.1f}s，推理 {pred.latency_ms:.0f}ms）")
    L.append(f"- 现价：{r.close if r.close is not None else '—'}　ATR14：{r.atr if r.atr is not None else '—'}")
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

    # ── 2. 价格计划
    L.append("## 二、价格计划（确定性计算：结构位优先 + ATR 兜底）")
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
    L.append("## 三、判据明细（Laya 原语输出）")
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
        a = pred.get(qid)
        if a and a.probabilities:
            L.append(f"**{qid} 概率分布**")
            L.append("")
            for k, v in sorted(a.probabilities.items(), key=lambda kv: -kv[1]):
                L.append(f"- {_DIRECTION_ZH.get(k, k)}：{v * 100:.1f}%")
            L.append("")

    # ── 4. 市场状态
    L.append("## 四、喂给模型的市场状态（可复核）")
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
</style>
</head>
<body><div class="wrap">
<h1>Laya 市场分析报告</h1>
<div class="meta">{symbol} · {timeframe} · {generated_at} · 设备 {device} · 推理 {latency_ms:.0f}ms</div>

{errors_html}

<h2>结论</h2>
<div class="card">
  <div class="big">{direction_zh}</div>
  <div class="muted">方向置信度 {direction_conf}</div>
  {struct_line}
  {signal_line}
</div>
{low_conf_html}

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


def render_html(r: LayaReport) -> str:
    """HTML 报告。极简未来主义：#F5F5F7 底 / #1D1D1F 字 / #FF5C1A 强调。"""
    pred = r.prediction
    dire = pred.get("方向")
    struct = pred.get("周期结构")
    sig = pred.get("信号有效")

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
        a = pred.get(qid)
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

    fmt = lambda v: "—" if v is None else str(v)  # noqa: E731
    return _HTML_TMPL.format(
        symbol=html.escape(r.symbol),
        timeframe=html.escape(r.timeframe),
        generated_at=html.escape(r.generated_at),
        device=html.escape(pred.device or "?"),
        latency_ms=pred.latency_ms,
        errors_html=errors_html,
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
