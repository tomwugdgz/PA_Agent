# -*- coding: utf-8 -*-
"""MT5 交易与回测包：Python 下单桥接 + 确定性回测 + MQL5 生成。"""
from pa_agent.mt5trading.backtest import BacktestParams, BacktestResult, run_backtest
from pa_agent.mt5trading.mt5_bridge import (
    AccountInfo,
    MT5BridgeError,
    OrderRequest,
    OrderResult,
    SymbolInfo,
    connect,
    send_order,
    shutdown,
)

__all__ = [
    "BacktestParams", "BacktestResult", "run_backtest",
    "AccountInfo", "MT5BridgeError", "OrderRequest", "OrderResult",
    "SymbolInfo", "connect", "send_order", "shutdown",
]
