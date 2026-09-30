# -*- coding: utf-8 -*-
"""经验库写入端：把成功的两阶段分析沉淀为可复用案例，并用后续 K 线自动判定盈亏。

背景（补齐「学习闭环」的写侧）
------------------------------
`ExperienceReader` 只读 `EXPERIENCE_DIR / <cycle_position> / {success,failure}_cases/`。
本模块负责三件事：

1. **落草稿**：两阶段分析成功后，把决策要素写成 pending 案例
   （`EXPERIENCE_DIR/_pending/`，Reader 不会读到，不会污染提示词）。
2. **判定**：等后续拿到了更新的 K 线，用「TP 先到还是 SL 先到」的确定性规则
   判定 success / failure，移动到对应目录。
3. **保底**：判定在 N 根内未触发 TP/SL 时，用 MFE/MAE 相对 1R 的关系裁决，
   规则确定性、可解释、写入案例本身供复核。

写入的 content 会被 prompt_assembler 按 `experience_max_chars_per_entry`
（默认 400 字符）截断，因此 content 刻意保持紧凑。
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PENDING_DIR_NAME = "_pending"
#: 超过这么多根收盘 K 线仍未触发 TP/SL → 用 MFE/MAE 裁决
ADJUDICATION_WINDOW_BARS = 50


def _pending_dir(experience_dir: Path) -> Path:
    return experience_dir / PENDING_DIR_NAME


def _case_dir(experience_dir: Path, cycle_position: str, case_type: str) -> Path:
    return experience_dir / cycle_position / f"{case_type}_cases"


def build_case_content(
    *,
    cycle_position: str,
    symbol: str,
    timeframe: str,
    direction: str,
    entry: float | None,
    stop: float | None,
    target: float | None,
    trade_confidence: int | None,
    zone: str,
    breakout_quality: str,
    outcome: dict[str, Any] | None,
) -> dict[str, Any]:
    """组装案例 content（会被 JSON 序列化后注入提示词，保持紧凑）。"""
    return {
        "symbol": symbol,
        "tf": timeframe,
        "cycle": cycle_position,
        "dir": direction,
        "entry": entry,
        "sl": stop,
        "tp": target,
        "conf": trade_confidence,
        "zone": zone,
        "breakout": breakout_quality,
        "outcome": outcome,   # None=未判定；否则 {status, bars, mfe_r, mae_r}
    }


def save_pending_case(
    *,
    experience_dir: Path,
    cycle_position: str,
    symbol: str,
    timeframe: str,
    direction: str,
    entry: float | None,
    stop: float | None,
    target: float | None,
    trade_confidence: int | None,
    zone: str,
    breakout_quality: str,
    analysis_ts_ms: int | None = None,
    source: str = "auto",
) -> Path | None:
    """把一次分析写成待判定案例。返回文件路径；任何失败只记日志返回 None。

    缺少 entry/stop（无单可判）时也落盘，但 `outcome.status` 恒为 `no_order`，
    永不晋升——此类案例只在补齐判定逻辑后人工处理。
    """
    try:
        experience_dir.mkdir(parents=True, exist_ok=True)
        pend = _pending_dir(experience_dir)
        pend.mkdir(exist_ok=True)

        ts_ms = analysis_ts_ms or int(time.time() * 1000)
        stamp = datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d_%H-%M-%S")
        safe_symbol = (symbol or "NA").replace("/", "-").replace("\\", "-")
        fname = f"{stamp}_{safe_symbol}_{timeframe}.json"

        content = build_case_content(
            cycle_position=cycle_position,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            entry=entry,
            stop=stop,
            target=target,
            trade_confidence=trade_confidence,
            zone=zone,
            breakout_quality=breakout_quality,
            outcome=None,
        )
        payload = {
            "filename": fname,
            "case_type": "pending",
            "cycle_position": cycle_position,
            "timestamp_ms": ts_ms,
            "symbol": symbol,
            "timeframe": timeframe,
            "source": source,
            "content": content,
        }
        path = pend / fname
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("经验案例已入待判定区: %s", path.name)
        return path
    except Exception as exc:  # noqa: BLE001 - 经验写入绝不能影响主流程
        logger.warning("写入经验案例失败: %s", exc)
        return None


# ── 判定 ─────────────────────────────────────────────────────────────────────


def adjudicate_pending(frame: Any, *, experience_dir: Path | None = None) -> int:
    """用当前 frame 里**更新的 K 线**裁决 pending 案例。返回晋升数。

    K 线顺序约定（PA_Agent 全局）：``frame.bars[0]`` 最新、``seq=1``，
    ``bars[-1]`` 最旧。案例时间戳落在某根收盘 K 线之后，就用「比它新的那些根」
    逐根扫描：先碰 TP 记 success，先碰 SL 记 failure。

    Raises:
        不抛——任何异常都吞掉并记日志，判定失败只意味着下次再试。
    """
    try:
        from pa_agent.config.paths import EXPERIENCE_DIR

        root = experience_dir or EXPERIENCE_DIR
        pend = _pending_dir(root)
        if not pend.is_dir():
            return 0
        bars = getattr(frame, "bars", None)
        symbol = str(getattr(frame, "symbol", "") or "")
        timeframe = str(getattr(frame, "timeframe", "") or "")
        if not bars or not symbol:
            return 0

        promoted = 0
        for path in sorted(pend.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("symbol") != symbol or payload.get("timeframe") != timeframe:
                continue
            if payload.get("outcome_status"):
                continue  # 已判定过（防重复晋升）
            ts_ms = payload.get("timestamp_ms") or 0
            newer = [b for b in bars if b.closed and getattr(b, "ts_open", 0) > ts_ms]
            if not newer:
                continue

            verdict = _judge(payload.get("content") or {}, newer)
            if verdict is None:
                continue
            if verdict.get("status") == "no_order":
                # 无单可判（模型没给入场/止损）：永不晋升，仅在原地打标记，
                # 防止每次重复扫描；留在 _pending 保证 Reader 永远读不到。
                _mark_no_order(path, payload, verdict)
                continue
            if _promote(root, pend, path, payload, verdict):
                promoted += 1
        return promoted
    except Exception as exc:  # noqa: BLE001
        logger.warning("经验案例判定失败: %s", exc)
        return 0


def _judge(content: dict[str, Any], newer_bars: list[Any]) -> dict[str, Any] | None:
    """对单个案例做 TP/SL 先后判定。K 线按时间升序传入（旧→新）。"""
    direction = str(content.get("dir") or "").lower()
    # 键名与 build_case_content 对齐：止损存的是 "sl"（不是 "stop"）
    entry, stop, target = content.get("entry"), content.get("sl"), content.get("tp")
    if direction not in ("long", "short", "buy", "sell") or entry is None or stop is None:
        return {"status": "no_order", "bars": 0, "mfe_r": None, "mae_r": None}
    is_long = direction in ("long", "buy")

    entry, stop = float(entry), float(stop)
    risk = abs(entry - stop)
    tp = float(target) if target is not None else None

    mfe_r, mae_r = 0.0, 0.0
    for i, b in enumerate(newer_bars[:ADJUDICATION_WINDOW_BARS], start=1):
        high, low = float(b.high), float(b.low)
        if is_long:
            fav = (high - entry) / risk if risk > 0 else 0.0
            adv = (entry - low) / risk if risk > 0 else 0.0
        else:
            fav = (entry - low) / risk if risk > 0 else 0.0
            adv = (high - entry) / risk if risk > 0 else 0.0
        mfe_r, mae_r = max(mfe_r, fav), max(mae_r, adv)
        hit_sl = low <= stop if is_long else high >= stop
        hit_tp = tp is not None and (high >= tp if is_long else low <= tp)
        if hit_tp and not hit_sl:
            return {"status": "tp", "bars": i, "mfe_r": round(mfe_r, 3), "mae_r": round(mae_r, 3)}
        if hit_sl and not hit_tp:
            return {"status": "sl", "bars": i, "mfe_r": round(mfe_r, 3), "mae_r": round(mae_r, 3)}
        if hit_tp and hit_sl:
            # 同根双触发，保守记失败（无法区分先后）
            return {"status": "ambiguous", "bars": i, "mfe_r": round(mfe_r, 3), "mae_r": round(mae_r, 3)}

    if len(newer_bars) >= ADJUDICATION_WINDOW_BARS:
        # 超时：MFE 先到 1R 视为成功，否则视为失败（确定性、可解释）
        status = "timeout_tp" if mfe_r >= 1.0 else "timeout_miss"
        return {"status": status, "bars": ADJUDICATION_WINDOW_BARS,
                "mfe_r": round(mfe_r, 3), "mae_r": round(mae_r, 3)}
    return None  # 数据还不够，继续等


_SUCCESS_STATUS = {"tp", "timeout_tp"}
_CASE_TYPE_BY_STATUS = {
    "tp": "success", "timeout_tp": "success",
    "sl": "failure", "timeout_miss": "failure", "ambiguous": "failure",
}


def _mark_no_order(path: Path, payload: dict[str, Any], verdict: dict[str, Any]) -> None:
    """给无单可判的案例原地打标记（不移动目录）。失败仅记日志。"""
    try:
        payload["content"]["outcome"] = verdict
        payload["outcome_status"] = "no_order"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("标记 no_order 案例失败 (%s): %s", path.name, exc)


def _promote(
    root: Path, pend: Path, path: Path, payload: dict[str, Any], verdict: dict[str, Any]
) -> bool:
    """把判定结果写回案例并移动到 success/failure 目录。"""
    try:
        status = verdict["status"]
        case_type = _CASE_TYPE_BY_STATUS.get(status, "failure")
        content = payload.get("content") or {}
        content["outcome"] = verdict
        payload["content"] = content
        payload["case_type"] = case_type
        payload["outcome_status"] = status
        payload["adjudicated_at_ms"] = int(time.time() * 1000)

        dest_dir = _case_dir(root, str(payload.get("cycle_position") or "unknown"), case_type)
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / path.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        path.unlink(missing_ok=True)
        logger.info("经验案例晋升: %s -> %s/%s", path.name, case_type, verdict)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("经验案例晋升失败 (%s): %s", path.name, exc)
        return False
