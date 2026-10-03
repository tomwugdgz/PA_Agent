# -*- coding: utf-8 -*-
"""Laya 报告生成管线：frame → 特征 → 状态 → Laya 推理 → 定价 → 报告对象。

这是 GUI「Laya 报告」按钮的唯一业务入口，也是后台 worker 调用的东西。
刻意不含任何 Qt 依赖，方便单元测试与脚本复用。
"""
from __future__ import annotations

import logging
import time
from typing import Any

from pa_agent.ai.laya_annotation import append_sample
from pa_agent.ai.laya_engine import LayaUnavailable, LayaEngine
from pa_agent.ai.laya_schema import LayaAnswer, LayaPrediction, build_questions, build_state
from pa_agent.ai.market_features import compute_simple_market_features
from pa_agent.report.laya_pricing import plan_long, plan_short
from pa_agent.report.laya_report import LayaReport
from pa_agent.util.price_tick import infer_price_tick_from_frame

logger = logging.getLogger(__name__)


def _resolve_device(cfg: Any) -> str:
    req = (getattr(cfg, "device", "auto") or "auto").strip().lower()
    if req in ("", "auto"):
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"
    return req


def _answers_from_raw(
    raw: dict[str, Any], *, min_confidence: float
) -> tuple[dict[str, LayaAnswer], list[str]]:
    """把 Laya 原始输出规整成 LayaAnswer 字典；解析失败的问题进 errors。

    实测 0.3.22 返回形状（tools/laya_smoke_test.py 探测确认）::

        {"answers": {
            "结构":   {"type": "choice", "choice": "trending_tr",
                       "probabilities": {...}, "confidence": 0.166,
                       "answer_confidence": 0.425, "action": {...}},
            "信号有效": {"type": "noul", "noul": 0.784,
                        "confidence": 0.784, "answer_confidence": 0.784}},
         "model": "laya-rl-agent", "usage": {...}}

    注意：`confidence` 是**校准后**置信度（工厂标定偏自信，微调前当参考），
    `answer_confidence` 是胜出选项的原始概率。
    """
    answers: dict[str, LayaAnswer] = {}
    errors: list[str] = []
    raw_answers = raw.get("answers")
    if not isinstance(raw_answers, dict):
        errors.append(f"Laya 返回缺 answers 字段：keys={sorted(raw.keys())}")
        return answers, errors

    for qid, spec in raw_answers.items():
        try:
            if not isinstance(spec, dict):
                errors.append(f"问题 {qid!r} 非预期结构：{type(spec).__name__}")
                continue
            qtype = str(spec.get("type") or questions_kind(qid) or "raw")
            if qtype == "choice":
                value = str(spec.get("choice", ""))
                probs = spec.get("probabilities") or {}
                conf = float(spec.get("confidence", 0.0))
            elif qtype == "noul":
                value = float(spec.get("noul", 0.0))
                probs = {}
                conf = float(spec.get("confidence", value))
            else:  # score 等
                value = spec.get("score", spec)
                probs = spec.get("probabilities") or {}
                conf = float(spec.get("confidence", 0.0))
            answers[qid] = LayaAnswer(
                qid=qid,
                kind=qtype,
                value=value,
                confidence=max(0.0, min(1.0, conf)),
                probabilities=probs if isinstance(probs, dict) else {},
                reliable=conf >= min_confidence,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"问题 {qid!r} 结果解析失败：{type(exc).__name__}: {exc}")
    return answers, errors


def questions_kind(qid: str) -> str | None:
    """查问题类型；未知 qid 返回 None（容错：Laya 不改问题集名）。"""
    from pa_agent.ai.laya_schema import build_questions

    for questions in (build_questions(), build_questions(with_breakout=False)):
        spec = questions.get(qid)
        if spec:
            return str(spec["type"])
    return None


def generate_report(frame: Any, settings: Any, progress: Any = None) -> LayaReport:
    """从一帧 K 线生成完整 Laya 报告。

    参数
    ----
    progress
        可选回调 ``progress(pct, msg)``，GUI 用来显示**真实**加载/推理进度。

    Raises:
        LayaUnavailable: 运行时 / 权重 / 推理失败（message 可直接展示给用户）。
        ValueError: 输入 frame 无效。
    """
    cfg = settings.laya
    bars = getattr(frame, "bars", None)
    if not bars:
        raise ValueError("K 线数据为空，无法生成 Laya 报告")

    close = float(bars[0].close)
    atr = None
    indicators = getattr(frame, "indicators", None)
    atr14 = getattr(indicators, "atr14", None) if indicators is not None else None
    if atr14:
        try:
            atr = float(atr14[0])
        except (TypeError, ValueError):
            atr = None

    if progress is not None:
        try:
            progress(3, f"准备 {len(bars)} 根 K 线特征…")
        except Exception:  # noqa: BLE001
            pass

    features = compute_simple_market_features(frame)

    # 校准文件（若存在且开关打开）自动挂载——置信度标定后才有参考价值
    cal = None
    if bool(getattr(cfg, "use_calibration", True)):
        from pa_agent.ai.laya_annotation import load_calibration_if_any

        cal = load_calibration_if_any(cfg.model_dir)

    engine = LayaEngine.get(
        model_dir=cfg.model_dir,
        subfolder=cfg.subfolder,
        device=_resolve_device(cfg),
        calibration=cal,
    )
    agent = engine.ensure_loaded(progress=progress)  # 校验 + 加载（幂等）
    del agent

    if progress is not None:
        try:
            progress(85, "构造问题与状态…")
        except Exception:  # noqa: BLE001
            pass

    questions = build_questions()
    state = build_state(
        symbol=str(getattr(frame, "symbol", "") or ""),
        timeframe=str(getattr(frame, "timeframe", "") or ""),
        features=features,
        atr=atr,
        close=close,
    )

    t0 = time.perf_counter()
    if progress is not None:
        try:
            progress(90, f"推理中（{len(questions)} 个问题）…")
        except Exception:  # noqa: BLE001
            pass
    raw = engine.predict(state, questions, lang="zh")
    latency_ms = (time.perf_counter() - t0) * 1000

    answers, errors = _answers_from_raw(
        raw, min_confidence=float(cfg.min_confidence)
    )
    prediction = LayaPrediction(
        answers=answers,
        usage=raw.get("usage") or {},
        latency_ms=latency_ms,
        load_ms=engine.load_ms,
        device=engine.device,
        calibrated=bool(getattr(engine, "calibrated", False)),
    )

    # ── 方向决定价格计划的主次（但两个方向都算，报告都展示）
    dire = answers.get("方向")
    tick = infer_price_tick_from_frame(frame)
    kwargs = dict(close=close, atr=atr, features=features, cfg=cfg, tick=tick)
    long_plan = plan_long(**kwargs)
    short_plan = plan_short(**kwargs)

    # 方向不可靠/明确反向时，在计划里补一条中文说明，报告里直接可见
    if dire and not dire.reliable:
        long_plan = _annotate(long_plan, f"方向置信度 {dire.confidence:.0%} 低于门槛，计划仅供参考")
        short_plan = _annotate(short_plan, f"方向置信度 {dire.confidence:.0%} 低于门槛，计划仅供参考")

    report = LayaReport(
        symbol=str(getattr(frame, "symbol", "") or ""),
        timeframe=str(getattr(frame, "timeframe", "") or ""),
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        prediction=prediction,
        long_plan=long_plan,
        short_plan=short_plan,
        close=close,
        atr=atr,
        supports=tuple(features.supports[:3]),
        resistances=tuple(features.resistances[:3]),
        state_text=state["body"],
        errors=tuple(errors),
    )

    # ── 三分支推导（多/空/观望 + 复合概率），失败绝不阻断报告
    try:
        from dataclasses import replace

        from pa_agent.report.laya_branches import derive_branches

        report = replace(
            report,
            branches=tuple(
                derive_branches(
                    close=close,
                    atr=atr,
                    features=features,
                    cfg=cfg,
                    answers=answers,
                    tick=tick,
                )
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Laya 分支推导失败，报告仍照常输出：%s", exc)

    # ── 标注数据收集（失败静默，绝不影响报告）
    if bool(getattr(cfg, "collect_annotations", False)):
        append_sample(
            state=state,
            questions=questions,
            answers={
                qid: {"kind": a.kind, "value": a.value, "confidence": a.confidence,
                      "probabilities": a.probabilities}
                for qid, a in answers.items()
            },
            context={
                "symbol": report.symbol,
                "timeframe": report.timeframe,
                "close": close,
                "atr": atr,
                "supports": list(report.supports),
                "resistances": list(report.resistances),
                "latency_ms": latency_ms,
                "device": engine.device,
            },
        )

    # ── 五层运行记录：数据 / 决策 / 风控（失败静默，绝不影响报告）
    _journal_report(report=report, frame=frame, cfg=cfg,
                    device=engine.device, latency_ms=latency_ms,
                    n_errors=len(errors))
    # ── 最近一次报告落盘（供 MT5 面板「导入分析」读取；失败静默）
    _persist_latest(report)
    if progress is not None:
        try:
            progress(100, "报告完成")
        except Exception:  # noqa: BLE001
            pass
    return report


def _persist_latest(report: LayaReport) -> None:
    """把最近一次 Laya 报告的价格计划写到 logs/laya_latest.json。

    MT5 面板与主窗口的分析是两条链路：面板只看得到 stage2_decision，
    看不到 Laya 报告——此文件就是两者之间的桥。任何异常吞掉。
    """
    try:
        import json as _json

        from pa_agent.config.paths import LOGS_DIR

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        dire = report.prediction.answers.get("方向")
        payload = {
            "generated_at": report.generated_at,
            "symbol": report.symbol,
            "timeframe": report.timeframe,
            "close": report.close,
            "atr": report.atr,
            "direction": (str(dire.value) if dire is not None else ""),
            "direction_confidence": (round(dire.confidence, 3)
                                     if dire is not None else None),
            "long_plan": {
                "actionable": report.long_plan.actionable,
                "reason": report.long_plan.reason,
                "entry": report.long_plan.entry,
                "stop": report.long_plan.stop,
                "target": report.long_plan.target,
                "rr": report.long_plan.rr_ratio,
            },
            "short_plan": {
                "actionable": report.short_plan.actionable,
                "reason": report.short_plan.reason,
                "entry": report.short_plan.entry,
                "stop": report.short_plan.stop,
                "target": report.short_plan.target,
                "rr": report.short_plan.rr_ratio,
            },
        }
        (LOGS_DIR / "laya_latest.json").write_text(
            _json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("laya_latest.json 写入失败（不影响报告）: %s", exc)


def _journal_report(
    *, report: LayaReport, frame: Any, cfg: Any,
    device: str, latency_ms: float, n_errors: int,
) -> None:
    """把本次报告的关键信息写入 journal 的 data / decision / risk 三层。

    任何异常都吞掉——journal 是纯外围旁路，主流程绝不因它失败。
    """
    try:
        from pa_agent.journal.layer_journal import (
            log_data, log_decision, log_risk,
        )

        sym, tf = report.symbol, report.timeframe
        # 数据层：输入快照
        log_data(
            symbol=sym, timeframe=tf,
            n_bars=len(getattr(frame, "bars", ()) or ()),
            close=report.close, atr=report.atr,
            device=device, latency_ms=latency_ms, n_errors=n_errors,
        )
        # 决策层：答题摘要 + 双向价格计划摘要
        answers = {
            qid: {"kind": a.kind, "value": a.value, "confidence": round(a.confidence, 3)}
            for qid, a in report.prediction.answers.items()
        }
        plans = {}
        for name, plan in (("long", report.long_plan), ("short", report.short_plan)):
            plans[name] = {
                "actionable": plan.actionable, "reason": plan.reason,
                "entry": plan.entry, "stop": plan.stop, "target": plan.target,
                "rr": plan.rr_ratio,
                "fallback": bool(plan.entry_fallback or plan.stop_fallback
                                 or plan.target_fallback),
            }
        log_decision(symbol=sym, timeframe=tf, source="laya",
                     answers=answers, plans=plans)
        # 风控层：计划告警 + 低置信度 + 最大盈亏比
        warnings: list[str] = []
        max_rr: float | None = None
        low_conf = False
        dire = report.prediction.answers.get("方向")
        if dire is not None and not dire.reliable:
            low_conf = True
        for plan in (report.long_plan, report.short_plan):
            warnings.extend(plan.notes)
            if plan.rr_ratio is not None:
                max_rr = plan.rr_ratio if max_rr is None else max(max_rr, plan.rr_ratio)
        log_risk(symbol=sym, timeframe=tf, warnings=warnings,
                 low_confidence=low_conf, max_rr=max_rr)
    except Exception as exc:  # noqa: BLE001
        logger.debug("journal 五层记录失败（不影响报告）: %s", exc)


def _annotate(plan: Any, note: str) -> Any:
    """给 PricePlan 追加一条 note（dataclass frozen，用替换法）。"""
    try:
        return type(plan)(**{**plan.__dict__, "notes": tuple(plan.notes) + (note,)})
    except Exception:  # noqa: BLE001
        return plan
