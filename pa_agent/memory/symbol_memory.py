"""品种记忆库：按「品种 + 周期」完全隔离的历次分析归档与结算。

为什么不用现有 journal
---------------------
``pa_agent/journal`` 是**按月分文件的全库流水**（``journal/2026-10/decision.jsonl``），
适合跨品种做全局审计；但本需求要的是"打开某个品种时调取它自己的历史"，
按月全库会让单品种查询退化成全表扫描，且不同品种的记录混在一行行里。

所以另建一套**按品种+周期隔离**的存储：

    experience/memory/<SYMBOL>/<TIMEFRAME>/
        sessions.jsonl     # 每次分析一条（append-only）
        outcomes.jsonl     # 每次结算一条（append-only）
        profile.json       # 该品种的滚动画像（覆盖写，��小）

设计约束
--------
* **纯外围**：所有写盘异常吞掉只记 debug 日志，绝不影响报告生成。
* **append-only**：``sessions`` / ``outcomes`` 只追加不改写，
  避免"重算历史"导致的数据漂移；``profile.json`` 是派生数据，可覆盖重算。
* **品种目录名净化**：品种名来自数据源，可能含 ``/``、空格等
  文件系统不友好的字符，落盘前统一转义（见 :func:`_safe_dirname`）。
* **结算滞后**：预测当时不知道结果，所以 ``sessions`` 先写、
  ``outcomes`` 后补（下次分析时结算上一条），两者用 ``session_id`` 关联。
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pa_agent.config.paths import EXPERIENCE_DIR

logger = logging.getLogger(__name__)

#: 记忆库根目录（挂在 experience 下，与 journal / laya_annotations 同级）
MEMORY_DIR: Path = EXPERIENCE_DIR / "memory"

#: 判定方向预测"正确"所需的最小后续涨跌幅（以 ATR 为单位）。
#: 低于这个幅度视为"没走出方向"，既不算对也不算错。
_DIRECTION_EPS_ATR = 0.5

#: 结算回看根数：向后取多少根 K 线判断结果。
_SETTLE_LOOKAHEAD = 10

#: 单次结算允许的最大跨度（毫秒）。超时说明行情已断档/换品种，
#: 强行结算会得出错误结论，直接标记为 ``stale`` 不计入准确率。
_SETTLE_MAX_AGE_MS = 7 * 24 * 3600 * 1000

_UNSAFE_CHARS = re.compile(r"[^\w\-.]+")


def _safe_dirname(raw: str, *, fallback: str = "UNKNOWN") -> str:
    """把品种/周期名转成安全的目录名。

    ``USD/CNH`` → ``USD_CNH``；``M15`` → ``M15``；
    空串或全非法字符 → ``fallback``。

    保留 ``\\w``（含中文）与 ``-`` ``.``，其余一律替换为 ``_``。

    ⚠️ 关键：**不做任何 strip**。曾经用 ``strip("._")`` 清理首尾，
    结果 ``_DBG`` 被变成 ``DBG``——写入用 ``_safe_dirname``、
    读取也用 ``_safe_dirname``，看似一致，但一旦规则在别处被改动
    （或手工建目录），就会读不到自己写的记录。保持"替换即返回"最安全：
    同一函数进出必然同名。
    """
    s = _UNSAFE_CHARS.sub("_", (raw or "").strip())
    return s or fallback


def symbol_dir(symbol: str, timeframe: str) -> Path:
    """某品种+周期的记忆库目录（不创建）。"""
    return MEMORY_DIR / _safe_dirname(symbol) / _safe_dirname(timeframe)


# ── 记录结构 ────────────────────────────────────────────────────────────────

@dataclass
class SessionRecord:
    """一次分析的完整归档（对应报告的一次生成）。"""

    session_id: str
    ts_ms: int
    ts: str# 本地时间字符串，便于人读
    symbol: str
    timeframe: str

    #: **决策时刻的行情时间**（最新那根 K 线的 ts_open）。
    #: 结算必须用它而不是 ``ts_ms``：``ts_ms`` 是本机墙钟，与行情时钟
    #: 可能差时区/差数小时（MT5 服务器时间实测偏 -3h），拿墙钟去比对
    #: K 线 ``ts_open`` 会得出"没有后续 K 线"的错误结论。
    #: 留 0 表示没有 frame，退回墙钟。
    bars_ts_ms: int = 0

    # ── 数据快照
    n_bars: int = 0
    close: float | None = None
    atr: float | None = None
    #: 最近一根 K 线的 OHLCV（记忆库的核心价值之一）
    ohlc: dict[str, float] = field(default_factory=dict)
    #: 最新均线值 {5: v, 10: v, 20: v, 60: v}，预热不足的周期不出现在字典里
    mas: dict[str, float] = field(default_factory=dict)
    ema20: float | None = None
    #: 均线排列形态：bullish（多头排列）/ bearish（空头排列）/ mixed
    ma_pattern: str = "unknown"
    #: 关键位
    supports: list[float] = field(default_factory=list)
    resistances: list[float] = field(default_factory=list)

    # ── 分析结果（Laya 答案）
    answers: dict[str, Any] = field(default_factory=dict)
    #: 三分支判断
    branches: list[dict[str, Any]] = field(default_factory=list)
    #: 主方向（bullish/bearish/neutral）
    direction: str | None = None
    direction_conf: float | None = None

    # ── 建议（价格计划）
    plans: dict[str, Any] = field(default_factory=dict)

    # ── 结果追踪（写入时为 None，结算后由 outcomes.jsonl 补）
    settled: bool = False


@dataclass
class OutcomeRecord:
    """对某次分析的结算：实际走势 vs 当时预测。"""

    session_id: str
    symbol: str
    timeframe: str
    open_ts_ms: int
    settle_ts_ms: int

    #: 预测方向与置信度
    predicted: str | None
    predicted_conf: float | None
    #: 实际方向（按 _DIRECTION_EPS_ATR 判定 bullish/bearish/flat）
    actual: str = "unknown"
    #: 预测方向与实际方向是否一致（actual=flat 时为 None，不计入准确率）
    direction_hit: bool | None = None

    #: 实际最高/最低价与触达情况
    high: float | None = None
    low: float | None = None
    #: 实际涨跌幅（以 ATR 为单位）
    move_atr: float | None = None
    #: 多空计划各自的触达结果
    long_result: str = "n/a"       # target_hit | stop_hit | neither | no_plan
    short_result: str = "n/a"

    #: 结算状态：ok | stale | no_bars | insufficient
    status: str = "ok"
    note: str = ""


@dataclass
class SymbolProfile:
    """某品种+周期的滚动画像（派生数据，可覆盖重算）。"""

    symbol: str
    timeframe: str
    updated_at: str = ""
    n_sessions: int = 0
    n_settled: int = 0
    #: 方向命中率（0~1），只统计方向明确的场次
    direction_accuracy: float | None = None
    #: 三分支里"观望"拿最高概率的占比——过高说明模型在该品种上
    #: 基本只会说"不做"，这是负面的能力信号
    wait_dominant_rate: float | None = None
    #: 方向分布（bullish/bearish/neutral 计数）
    direction_counts: dict[str, int] = field(default_factory=dict)
    #: 平均 ATR 与常见价格区间，供下次快速判断波动量级
    atr_avg: float | None = None
    price_min: float | None = None
    price_max: float | None = None
    #: 历史上出现过的支撑/阻力，取最常出现的前 3 个
    common_supports: list[float] = field(default_factory=list)
    common_resistances: list[float] = field(default_factory=list)
    #: 最近 10 次的结算摘要，供报告里做"上次预测准不准"提示
    recent_outcomes: list[dict[str, Any]] = field(default_factory=list)


# ── 写入 ────────────────────────────────────────────────────────────────────

def _append_jsonl(path: Path, payload: dict[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return True


def _ma_pattern(mas: dict[str, float]) -> str:
    """判定均线排列形态。

    只用 5/10/20 三条判方向（MA60 过滤太慢，不参与短周期排列判断）；
    数据不足时返回 unknown。
    """
    need = (5, 10, 20)
    if any(p not in mas for p in need):
        return "unknown"
    m5, m10, m20 = (mas[p] for p in need)
    if m5 > m10 > m20:
        return "bullish"
    if m5 < m10 < m20:
        return "bearish"
    return "mixed"


def save_session(
    *,
    symbol: str,
    timeframe: str,
    report: Any,
    frame: Any = None,
) -> str | None:
    """把一次分析归档到该品种的记忆库，返回 session_id。

    永不抛异常——任何失败只记 debug 日志并返回 None。
    """
    try:
        return _save_session_impl(symbol, timeframe, report, frame)
    except Exception as exc:  # noqa: BLE001
        logger.debug("记忆库写入失败（不影响报告）: %s", exc)
        return None


def _save_session_impl(
    symbol: str, timeframe: str, report: Any, frame: Any
) -> str:
    now_ms = int(time.time() * 1000)
    sid = uuid.uuid4().hex[:16]

    # ── 数据快照：均线 + OHLC ──
    mas: dict[str, float] = {}
    ohlc: dict[str, float] = {}
    ema20: float | None = None
    n_bars = 0
    bars_ts_ms = 0
    if frame is not None:
        n_bars = len(getattr(frame, "bars", ()) or ())
        ind = getattr(frame, "indicators", None)
        if ind is not None:
            for p, seq in (getattr(ind, "mas", {}) or {}).items():
                if seq:
                    try:
                        v = float(seq[0])
                    except (TypeError, ValueError):
                        continue
                    if not math.isnan(v):
                        mas[str(p)] = v
            ema20 = ind.ma(20) if hasattr(ind, "ma") else None
        bars = getattr(frame, "bars", ()) or ()
        if bars:
            b0 = bars[0]
            ohlc = {
                "open": float(b0.open), "high": float(b0.high),
                "low": float(b0.low), "close": float(b0.close),
                "volume": float(b0.volume),
            }
            try:
                bars_ts_ms = int(b0.ts_open)
            except (TypeError, ValueError, AttributeError):
                bars_ts_ms = 0

    # ── 分析结果 ──
    answers: dict[str, Any] = {}
    direction = None
    direction_conf = None
    pred = getattr(report, "prediction", None)
    ans_map = getattr(pred, "answers", {}) or {}
    for qid, a in ans_map.items():
        try:
            answers[qid] = {
                "kind": a.kind, "value": _jsonable(a.value),
                "confidence": round(float(a.confidence), 4),
            }
        except Exception:  # noqa: BLE001
            continue
    d_ans = ans_map.get("方向")
    if d_ans is not None:
        direction = str(getattr(d_ans, "value", "") or "") or None
        try:
            direction_conf = round(float(d_ans.confidence), 4)
        except Exception:  # noqa: BLE001
            direction_conf = None

    # ── 三分支 ──
    branches: list[dict[str, Any]] = []
    for b in (getattr(report, "branches", ()) or ()):
        if isinstance(b, dict):
            branches.append(b)
            continue
        plan = getattr(b, "plan", None)
        branches.append({
            "key": getattr(b, "key", ""),
            "label": getattr(b, "label", ""),
            "probability": round(float(getattr(b, "probability", 0.0)), 4),
            "actionable": bool(getattr(b, "actionable", False)),
            "low_prob": bool(getattr(b, "low_prob", False)),
            "entry": _num(getattr(plan, "entry", None)),
            "stop": _num(getattr(plan, "stop", None)),
            "target": _num(getattr(plan, "target", None)),
            "rr": _num(getattr(plan, "rr_ratio", None)),
        })

    # ── 价格计划（建议）──
    plans: dict[str, Any] = {}
    for name, plan in (
        ("long", getattr(report, "long_plan", None)),
        ("short", getattr(report, "short_plan", None)),
    ):
        if plan is None:
            continue
        plans[name] = {
            "actionable": bool(getattr(plan, "actionable", False)),
            "reason": getattr(plan, "reason", ""),
            "entry": _num(getattr(plan, "entry", None)),
            "stop": _num(getattr(plan, "stop", None)),
            "target": _num(getattr(plan, "target", None)),
            "rr": _num(getattr(plan, "rr_ratio", None)),
            "notes": list(getattr(plan, "notes", ()) or ()),
        }

    rec = SessionRecord(
        session_id=sid,
        ts_ms=now_ms,
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        symbol=symbol,
        timeframe=timeframe,
        bars_ts_ms=bars_ts_ms or now_ms,
        n_bars=n_bars,
        close=_num(getattr(report, "close", None)),
        atr=_num(getattr(report, "atr", None)),
        ohlc=ohlc,
        mas=mas,
        ema20=_num(ema20),
        ma_pattern=_ma_pattern({int(k): v for k, v in mas.items()}),
        supports=[_num(x) for x in (getattr(report, "supports", ()) or ()) if _num(x) is not None],
        resistances=[_num(x) for x in (getattr(report, "resistances", ()) or ()) if _num(x) is not None],
        answers=answers,
        branches=branches,
        direction=direction,
        direction_conf=direction_conf,
        plans=plans,
    )

    d = symbol_dir(symbol, timeframe)
    _append_jsonl(d / "sessions.jsonl", asdict(rec))
    return sid


# ── 读取 ────────────────────────────────────────────────────────────────────

def list_sessions(
    symbol: str, timeframe: str, *, limit: int = 50
) -> list[dict[str, Any]]:
    """列出该品种最近 N 次分析（按时间倒序）。文件不存在返回空表。"""
    return _read_recent(symbol_dir(symbol, timeframe) / "sessions.jsonl", limit)


def list_outcomes(
    symbol: str, timeframe: str, *, limit: int = 50
) -> list[dict[str, Any]]:
    """列出该品种最近 N 次结算（按时间倒序）。"""
    return _read_recent(symbol_dir(symbol, timeframe) / "outcomes.jsonl", limit)


def _read_recent(path: Path, limit: int) -> list[dict[str, Any]]:
    try:
        if not path.is_file():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for raw in reversed(lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except ValueError:
            continue
        if len(out) >= limit:
            break
    return out


def load_profile(symbol: str, timeframe: str) -> SymbolProfile:
    """读取该品种画像；不存在返回空画像。"""
    p = symbol_dir(symbol, timeframe) / "profile.json"
    try:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            known = {f for f in SymbolProfile.__dataclass_fields__}
            return SymbolProfile(**{k: v for k, v in data.items() if k in known})
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("画像读取失败: %s", exc)
    return SymbolProfile(symbol=symbol, timeframe=timeframe)


def all_symbols() -> list[tuple[str, str, int]]:
    """列出记忆库里所有（品种, 周期, 会话数）。

    供 GUI 做"有哪些品种的记忆库"选择列表。
    """
    try:
        if not MEMORY_DIR.is_dir():
            return []
        out: list[tuple[str, str, int]] = []
        for sym_dir in sorted(MEMORY_DIR.iterdir()):
            if not sym_dir.is_dir():
                continue
            for tf_dir in sorted(sym_dir.iterdir()):
                if not tf_dir.is_dir():
                    continue
                f = tf_dir / "sessions.jsonl"
                n = 0
                if f.is_file():
                    n = sum(1 for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
                out.append((sym_dir.name, tf_dir.name, n))
        return out
    except OSError as exc:
        logger.debug("枚举品种失败: %s", exc)
        return []


# ── 工具 ────────────────────────────────────────────────────────────────────

def _num(v: Any) -> float | None:
    """转 float；None / nan / 不可转都返回 None。"""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _jsonable(v: Any) -> Any:
    """把 Laya 答案值转成可 JSON 序列化的形式。"""
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    f = _num(v)
    return f if f is not None else str(v)
