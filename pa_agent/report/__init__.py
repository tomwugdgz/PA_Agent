# -*- coding: utf-8 -*-
"""报告包：Laya 本地结构化报告（与 DeepSeek 两阶段分析解耦）。"""
from pa_agent.report.laya_pricing import PricePlan, plan_long, plan_short
from pa_agent.report.laya_report import (
    LayaReport,
    render_html,
    render_markdown,
    save_pair,
)

__all__ = [
    "PricePlan",
    "plan_long",
    "plan_short",
    "LayaReport",
    "render_html",
    "render_markdown",
    "save_pair",
]
