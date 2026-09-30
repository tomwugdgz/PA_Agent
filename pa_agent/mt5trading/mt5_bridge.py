# -*- coding: utf-8 -*-
"""MT5 终端桥接：连接检查、账户/品种信息、下单（市价/限价/突破）。

安全边界（刻意的架构决定）
--------------------------
* 本模块**只提供单次、显式调用**的下单函数；没有任何轮询/策略循环会调用它。
* 所有下单调用必须来自 GUI 的确认弹窗之后——调用方（mt5_panel）负责确认，
  本模块再做最后一道机器校验（点差、手数范围、止损方向）。
* `MetaTrader5` 包缺失或终端未运行时抛 `MT5BridgeError`，message 面向用户。

MT5 Python API 事实（5.0.45 实测行为）：
* `initialize()` 不带参数 = 附加到**已登录的运行中终端**（推荐路径）。
* `order_send` 返回 dict 或 None（None = 结果结构异常，要查 last_error）。
* retcode 10009=TRADE_RETCODE_DONE, 10008=PLACED（挂单受理）。
* 填充策略各经纪商不一：FOK → IOC → RETURN 依次降级重试。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: order_send 成功 retcode
_RETCODE_DONE = 10009
_RETCODE_PLACED = 10008
#: 填充策略降级顺序（MT5 枚举：ORDER_FILLING_FOK=0, IOC=1, RETURN=2）
_FILLING_CANDIDATES = (0, 1, 2)


class MT5BridgeError(RuntimeError):
    """MT5 桥接失败。message 面向终端用户，可直接展示。"""


def _fmt_err() -> str:
    try:
        import MetaTrader5 as mt5

        err = mt5.last_error()
        return f"{err[0]}: {err[1]}" if err and err[0] else "未知错误"
    except Exception:  # noqa: BLE001
        return "MetaTrader5 不可用"


@dataclass(frozen=True)
class AccountInfo:
    login: int
    server: str
    currency: str
    balance: float
    equity: float
    leverage: int


@dataclass(frozen=True)
class SymbolInfo:
    name: str
    digits: int
    point: float
    spread_points: int
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level_points: int


def check_package() -> str | None:
    """MetaTrader5 包可用性预检。None=可用。"""
    try:
        import MetaTrader5  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return (
            f"MetaTrader5 Python 包不可用：{exc}。"
            "该包仅支持 Windows + Python 3.12（本程序已装），请检查运行环境。"
        )
    return None


def connect(terminal_path: str = "") -> AccountInfo:
    """附加到运行中的 MT5 终端并返回账户信息。

    Raises:
        MT5BridgeError: 包缺失 / 终端未运行 / 未登录。
    """
    reason = check_package()
    if reason:
        raise MT5BridgeError(reason)
    import MetaTrader5 as mt5

    kwargs: dict[str, Any] = {}
    if terminal_path.strip():
        kwargs["path"] = terminal_path.strip()
    if not mt5.initialize(**kwargs):
        raise MT5BridgeError(f"无法连接 MT5 终端（{_fmt_err()}）。请先启动并登录 MT5。")
    acc = mt5.account_info()
    if acc is None:
        mt5.shutdown()
        raise MT5BridgeError(f"MT5 已连接但未登录账户（{_fmt_err()}）。")
    return AccountInfo(
        login=int(acc.login),
        server=str(acc.server),
        currency=str(acc.currency),
        balance=float(acc.balance),
        equity=float(acc.equity),
        leverage=int(acc.leverage),
    )


def shutdown() -> None:
    """释放终端连接（幂等）。"""
    try:
        import MetaTrader5 as mt5

        mt5.shutdown()
    except Exception:  # noqa: BLE001
        pass


def get_symbol_info(symbol: str) -> SymbolInfo:
    """读取品种规格。自动 select 以确保可见。"""
    import MetaTrader5 as mt5

    if not mt5.symbol_select(symbol, False):
        raise MT5BridgeError(f"品种 {symbol} 在 MT5 中不可见（{_fmt_err()}）。")
    info = mt5.symbol_info(symbol)
    if info is None:
        raise MT5BridgeError(f"无法读取品种信息 {symbol}（{_fmt_err()}）。")
    return SymbolInfo(
        name=str(info.name),
        digits=int(info.digits),
        point=float(info.point),
        spread_points=int(getattr(info, "spread", 0)),
        volume_min=float(info.volume_min),
        volume_max=float(info.volume_max),
        volume_step=float(info.volume_step),
        stops_level_points=int(getattr(info, "trade_stops_level", 0)),
    )


def normalize_lot(lot: float, sym: SymbolInfo) -> float:
    """把手数对齐到品种的 min/max/step。"""
    step = sym.volume_step or 0.01
    steps = round((lot - sym.volume_min) / step)
    lot = sym.volume_min + steps * step
    return max(sym.volume_min, min(sym.volume_max, round(lot, 8)))


@dataclass(frozen=True)
class OrderRequest:
    """一次下单请求（GUI 确认后由 bridge 执行）。"""

    symbol: str
    direction: str            # long | short
    order_kind: str           # market | limit | stop
    entry: float | None       # limit/stop 必填；market 用 None
    stop_loss: float
    take_profit: float | None = None
    take_profit2: float | None = None   # 仅记录，MT5 单笔只支持一个 TP
    lot: float = 0.01
    comment: str = "PA_Agent"


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    retcode: int | None
    order_ticket: int | None
    message: str              # 成功/失败的人类可读说明


def _filling_name(v: int) -> str:
    return {0: "FOK", 1: "IOC", 2: "RETURN"}.get(v, str(v))


def _send_order_impl(req: OrderRequest, *, cfg: Any) -> OrderResult:
    """执行一次下单。最后一道机器校验在此完成。

    Raises:
        MT5BridgeError: 连接/品种问题；**校验不通过返回 OrderResult(ok=False)**，
            不抛异常（让 GUI 统一展示原因）。
    """
    import MetaTrader5 as mt5

    connect(cfg.terminal_path)  # 幂等：已连接时无副作用
    try:
        sym = get_symbol_info(req.symbol)

        if cfg.max_spread_points and sym.spread_points > cfg.max_spread_points:
            return OrderResult(
                False, None, None,
                f"点差 {sym.spread_points} 超过上限 {cfg.max_spread_points} point，拒绝下单",
            )

        lot = normalize_lot(req.lot, sym)
        tick = mt5.symbol_info_tick(req.symbol)
        if tick is None:
            return OrderResult(False, None, None, f"无 {req.symbol} 行情报价（{_fmt_err()}）")

        if req.direction == "long":
            otype_map = {"market": mt5.ORDER_TYPE_BUY,
                         "limit": mt5.ORDER_TYPE_BUY_LIMIT,
                         "stop": mt5.ORDER_TYPE_BUY_STOP}
            ref_price = tick.ask
        else:
            otype_map = {"market": mt5.ORDER_TYPE_SELL,
                         "limit": mt5.ORDER_TYPE_SELL_LIMIT,
                         "stop": mt5.ORDER_TYPE_SELL_STOP}
            ref_price = tick.bid

        if req.order_kind not in otype_map:
            return OrderResult(False, None, None, f"未知订单类型 {req.order_kind!r}")
        otype = otype_map[req.order_kind]

        price = ref_price if req.order_kind == "market" else float(req.entry)
        if price is None or price <= 0:
            return OrderResult(False, None, None, "挂单缺少有效入场价")

        # ── 止损方向校验（交易所层面止盈同向）
        if req.stop_loss is None or req.stop_loss <= 0:
            return OrderResult(False, None, None, "缺少止损价")
        if req.direction == "long" and not (req.stop_loss < price):
            return OrderResult(False, None, None, f"多头止损 {req.stop_loss} 必须低于入场 {price}")
        if req.direction == "short" and not (req.stop_loss > price):
            return OrderResult(False, None, None, f"空头止损 {req.stop_loss} 必须高于入场 {price}")

        base: dict[str, Any] = {
            "action": (mt5.TRADE_ACTION_DEAL if req.order_kind == "market"
                       else mt5.TRADE_ACTION_PENDING),
            "symbol": req.symbol,
            "volume": lot,
            "type": otype,
            "price": price,
            "sl": float(req.stop_loss),
            "deviation": 20,
            "magic": int(cfg.magic),
            "comment": req.comment[:31],   # MT5 注释上限 31 字符
            "type_time": mt5.ORDER_TIME_GTC,
        }
        if req.take_profit:
            base["tp"] = float(req.take_profit)

        last_code, last_msg = None, ""
        for filling in _FILLING_CANDIDATES:
            request = dict(base, type_filling=filling)
            result = mt5.order_send(request)
            if result is None:
                last_code, last_msg = None, _fmt_err()
                continue
            last_code = int(result.retcode)
            if last_code in (_RETCODE_DONE, _RETCODE_PLACED):
                verb = "成交" if last_code == _RETCODE_DONE else "已受理"
                return OrderResult(
                    True, last_code, int(getattr(result, "order", 0) or 0),
                    f"{req.order_kind} 单{verb}：{req.symbol} {req.direction} "
                    f"{lot} 手 @ {price}，SL {req.stop_loss}"
                    + (f"，TP {req.take_profit}" if req.take_profit else "")
                    + f"（{_filling_name(filling)}，票号 #{result.order}）",
                )
            last_msg = str(getattr(result, "comment", "") or result.retcode)
            # 10030=Unsupported filling mode → 换下一种填充策略重试
            if last_code != 10030:
                break

        return OrderResult(False, last_code, None, f"下单失败 retcode={last_code}：{last_msg}")
    finally:
        pass  # 保持连接供后续查询；进程退出由 shutdown() 兜底


def send_order(req: OrderRequest, *, cfg: Any) -> OrderResult:
    """``_send_order_impl`` 的记录包装：执行层 journal 落盘 + 原样返回。

    成功 / 被拒 / 抛异常三种结局都记（异常记为 order_rejected + message），
    journal 自身失败被吞掉，绝不影响下单主流程。
    """
    try:
        result = _send_order_impl(req, cfg=cfg)
    except MT5BridgeError as exc:
        _journal_execution(req, ok=False, retcode=None, ticket=None,
                           message=str(exc))
        raise
    except Exception as exc:  # noqa: BLE001 - 非 MT5BridgeError 的意外异常也记
        _journal_execution(req, ok=False, retcode=None, ticket=None,
                           message=f"意外异常：{exc}")
        raise
    _journal_execution(req, ok=result.ok, retcode=result.retcode,
                       ticket=result.order_ticket, message=result.message)
    return result


def _journal_execution(
    req: OrderRequest, *, ok: bool, retcode: int | None,
    ticket: int | None, message: str,
) -> None:
    """执行层记录（旁路，失败静默）。"""
    try:
        from pa_agent.journal.layer_journal import log_execution

        log_execution(
            symbol=req.symbol, ok=ok, retcode=retcode,
            direction=req.direction, order_kind=req.order_kind,
            entry=req.entry, stop_loss=req.stop_loss,
            take_profit=req.take_profit, lot=req.lot,
            ticket=ticket, message=message,
        )
    except Exception:  # noqa: BLE001
        pass
