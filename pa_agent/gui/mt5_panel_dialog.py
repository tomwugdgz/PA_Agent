# -*- coding: utf-8 -*-
"""「MT5 交易」面板：连接 / 下单 / 生成 MQL5 / 回测，四合一。

安全设计
--------
* 下单必须满足三道门：settings.mt5trading.enabled=True、
  confirm_required=True 时的 QMessageBox 确认、bridge 内部机器校验。
* 回测在 QThread 中运行（几百次特征计算，秒级），UI 不冻结。
* MT5 未连接/包缺失时，下单与连接给出可直接阅读的错误，其余功能区不受影响。
"""
from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class BacktestWorker(QThread):
    """后台跑确定性回测。"""

    ready = pyqtSignal(object)    # BacktestResult
    failed = pyqtSignal(str)

    def __init__(self, frame: Any, params: Any, parent: Any = None) -> None:
        super().__init__(parent)
        self._frame = frame
        self._params = params

    def run(self) -> None:
        try:
            from pa_agent.mt5trading.backtest import run_backtest

            self.ready.emit(run_backtest(self._frame, self._params))
        except Exception as exc:  # noqa: BLE001
            logger.warning("BacktestWorker failed: %s", exc)
            self.failed.emit(str(exc))


class MT5PanelDialog(QDialog):
    """MT5 四合一面板（模态）。"""

    def __init__(self, frame: Any, record: Any, settings: Any, parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MT5 交易 · 回测 · MQL5")
        self.resize(860, 640)
        self._frame = frame
        self._record = record
        self._settings = settings
        self._cfg = getattr(settings, "mt5trading", None)
        self._bt_worker: BacktestWorker | None = None
        self._connected: bool = False

        root = QVBoxLayout(self)
        self._acct_label = QLabel("MT5：未连接。请先启动并登录 MT5 终端，再点「连接」。")
        self._acct_label.setObjectName("mutedLabel")
        self._acct_label.setWordWrap(True)
        root.addWidget(self._acct_label)

        tabs = QTabWidget()
        tabs.addTab(self._build_trade_tab(), "下单（当前决策）")
        tabs.addTab(self._build_mql5_tab(), "生成 MQL5")
        tabs.addTab(self._build_backtest_tab(), "回测")
        root.addWidget(tabs, 1)

        self._status = QLabel("")
        self._status.setObjectName("mutedLabel")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

    # ── Tab 1：下单 ───────────────────────────────────────────────────────────

    def _build_trade_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        self._decision_text = QTextBrowser()
        self._decision_text.setMaximumHeight(200)
        v.addWidget(self._decision_text)
        self._show_decision()

        row = QHBoxLayout()
        self._connect_btn = QPushButton("连接 MT5")
        self._connect_btn.clicked.connect(self._on_connect)
        self._send_btn = QPushButton("确认后向 MT5 下单")
        self._send_btn.clicked.connect(self._on_send_order)
        if self._cfg is None or not bool(getattr(self._cfg, "enabled", False)):
            self._send_btn.setEnabled(False)
            self._send_btn.setToolTip(
                "已禁用：config/settings.json → mt5trading.enabled 设为 true 后可用"
            )
        row.addWidget(self._connect_btn)
        row.addWidget(self._send_btn)
        row.addStretch()
        v.addLayout(row)
        v.addStretch()
        return w

    def _decision(self) -> dict[str, Any]:
        s2 = getattr(self._record, "stage2_decision", None) if self._record else None
        return s2 if isinstance(s2, dict) else {}

    def _show_decision(self) -> None:
        d = self._decision()
        if not d:
            self._decision_text.setHtml(
                "<p>当前没有 AI 分析决策。请先在主窗口「提交分析」成功后再来下单。</p>"
            )
            return
        if str(d.get("order_type") or "") not in ("限价单", "突破单", "市价单"):
            self._decision_text.setHtml(
                "<p>最近一次 AI 分析未给出可执行订单（no_order），无法下单。</p>"
            )
            return
        rows = [
            ("方向", d.get("order_direction")), ("订单类型", d.get("order_type")),
            ("入场", d.get("entry_price")), ("止损", d.get("stop_loss_price")),
            ("止盈 TP1", d.get("take_profit_price")),
            ("止盈 TP2（记录用）", d.get("take_profit_price_2")),
            ("置信度", d.get("trade_confidence")),
            ("入场依据", d.get("entry_rule")),
            ("失效条件", d.get("invalidation_condition")),
        ]
        html = "".join(
            f"<p><b>{k}</b>：{'' if v is None else v}</p>" for k, v in rows
        )
        self._decision_text.setHtml(html)

    def _on_connect(self) -> None:
        try:
            from pa_agent.mt5trading.mt5_bridge import connect

            acc = connect(getattr(self._cfg, "terminal_path", "") or "")
            self._connected = True
            self._acct_label.setText(
                f"已连接：账户 {acc.login} @ {acc.server} · 余额 {acc.balance:.2f} "
                f"{acc.currency} · 净值 {acc.equity:.2f} · 杠杆 1:{acc.leverage}"
            )
        except Exception as exc:  # noqa: BLE001
            self._connected = False
            self._acct_label.setText(f"连接失败：{exc}")

    def _on_send_order(self) -> None:
        d = self._decision()
        if not d:
            QMessageBox.information(self, "MT5 下单", "没有可执行的 AI 决策，请先「提交分析」。")
            return
        if self._cfg is not None and bool(getattr(self._cfg, "confirm_required", True)):
            ok = QMessageBox.question(
                self, "确认下单",
                "即将向 MT5 发送真实订单：\n\n"
                f"方向：{d.get('order_direction')}　类型：{d.get('order_type')}\n"
                f"入场：{d.get('entry_price')}　止损：{d.get('stop_loss_price')}\n"
                f"止盈：{d.get('take_profit_price')}\n\n确认继续？",
            ) == QMessageBox.StandardButton.Yes
            if not ok:
                return
        try:
            from pa_agent.mt5trading.mt5_bridge import OrderRequest, send_order

            kind_map = {"市价单": "market", "限价单": "limit", "突破单": "stop"}
            meta = getattr(self._record, "meta", None)
            req = OrderRequest(
                symbol=str(getattr(meta, "symbol", "") or ""),
                direction=str(d.get("order_direction") or "").lower(),
                order_kind=kind_map.get(str(d.get("order_type") or ""), "market"),
                entry=float(d["entry_price"]) if d.get("entry_price") else None,
                stop_loss=float(d["stop_loss_price"]),
                take_profit=float(d["take_profit_price"]) if d.get("take_profit_price") else None,
                lot=float(getattr(self._cfg, "default_lot", 0.01)),
                comment="PA_Agent",
            )
            result = send_order(req, cfg=self._cfg)
            self._status.setText(result.message)
            if result.ok:
                QMessageBox.information(self, "MT5 下单", result.message)
            else:
                QMessageBox.warning(self, "MT5 下单", result.message)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "MT5 下单", str(exc))

    # ── Tab 2：生成 MQL5 ─────────────────────────────────────────────────────

    def _build_mql5_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        info = QLabel(
            "两份模板：\n"
            "① 决策执行 EA —— 把当前 AI 决策（方向/入场/止损/止盈）写成参数默认值，"
            "挂到图表自动执行这一单；\n"
            "② 策略回测 EA —— 完整结构位+ATR 规则（与内置回测同口径），"
            "供 MT5 策略测试器回测与实盘。\n\n"
            "生成后请用 MetaEditor 打开编译（F7），先在策略测试器验证再上实盘。"
        )
        info.setWordWrap(True)
        v.addWidget(info)

        row = QHBoxLayout()
        btn1 = QPushButton("① 生成决策执行 EA")
        btn1.clicked.connect(self._gen_decision_ea)
        btn2 = QPushButton("② 生成策略回测 EA")
        btn2.clicked.connect(self._gen_strategy_ea)
        btn3 = QPushButton("打开输出目录")
        btn3.clicked.connect(self._open_mql5_dir)
        for b in (btn1, btn2, btn3):
            row.addWidget(b)
        row.addStretch()
        v.addLayout(row)
        v.addStretch()
        return w

    def _mql5_dir(self):
        from pa_agent.config.paths import LOGS_DIR

        return LOGS_DIR / "mql5"

    def _gen_decision_ea(self) -> None:
        d = self._decision()
        if not d:
            QMessageBox.information(self, "生成 MQL5", "没有 AI 决策可导出，请先「提交分析」。")
            return
        try:
            from pa_agent.mt5trading.mq5_generator import generate_decision_ea, save_mq5

            meta = getattr(self._record, "meta", None)
            src = generate_decision_ea(
                decision=d,
                symbol=str(getattr(meta, "symbol", "") or ""),
                timeframe=str(getattr(meta, "timeframe", "") or ""),
                magic=int(getattr(self._cfg, "magic", 20260930)),
                lot=float(getattr(self._cfg, "default_lot", 0.01)),
                max_spread_points=int(getattr(self._cfg, "max_spread_points", 0)),
                expiry_bars=int(getattr(self._cfg, "pending_expiry_bars", 12)),
            )
            path = save_mq5(src, self._mql5_dir(), "PA_DecisionEA")
            self._status.setText(f"已生成：{path}")
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "生成 MQL5", str(exc))

    def _gen_strategy_ea(self) -> None:
        try:
            from pa_agent.mt5trading.mq5_generator import generate_strategy_ea, save_mq5

            src = generate_strategy_ea(
                magic=int(getattr(self._cfg, "magic", 20260930)) if self._cfg else 20260930,
                lot=float(getattr(self._cfg, "default_lot", 0.01)) if self._cfg else 0.01,
                max_spread_points=int(getattr(self._cfg, "max_spread_points", 0)) if self._cfg else 0,
                lookback=int(getattr(self._cfg, "backtest_lookback", 100)) if self._cfg else 100,
                timeout_bars=int(getattr(self._cfg, "backtest_timeout_bars", 50)) if self._cfg else 50,
            )
            path = save_mq5(src, self._mql5_dir(), "PA_StrategyEA")
            self._status.setText(f"已生成：{path}")
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "生成 MQL5", str(exc))

    def _open_mql5_dir(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._mql5_dir())))

    # ── Tab 3：回测 ───────────────────────────────────────────────────────────

    def _build_backtest_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        cfg = self._cfg

        grid = QHBoxLayout()
        self._sp_lookback = QSpinBox(); self._sp_lookback.setRange(20, 2000)
        self._sp_lookback.setValue(int(getattr(cfg, "backtest_lookback", 100)) if cfg else 100)
        self._sp_stop = QDoubleSpinBox(); self._sp_stop.setRange(0.05, 5.0); self._sp_stop.setSingleStep(0.05)
        self._sp_stop.setValue(0.25)
        self._sp_target = QDoubleSpinBox(); self._sp_target.setRange(0.5, 10.0); self._sp_target.setSingleStep(0.5)
        self._sp_target.setValue(2.0)
        self._sp_cost = QSpinBox(); self._sp_cost.setRange(0, 500)
        self._sp_cost.setValue(int(getattr(cfg, "backtest_cost_points", 10)) if cfg else 10)
        for label, sp in (("回看(根)", self._sp_lookback), ("止损缓冲×ATR", self._sp_stop),
                          ("目标×R", self._sp_target), ("成本(点/边)", self._sp_cost)):
            box = QWidget(); hb = QHBoxLayout(box); hb.setContentsMargins(0, 0, 0, 0)
            hb.addWidget(QLabel(label)); hb.addWidget(sp)
            grid.addWidget(box)
        v.addLayout(grid)

        self._cb_limit = QCheckBox("限价单（回踩）"); self._cb_limit.setChecked(True)
        self._cb_break = QCheckBox("突破单"); self._cb_break.setChecked(True)
        row = QHBoxLayout(); row.addWidget(self._cb_limit); row.addWidget(self._cb_break); row.addStretch()
        v.addLayout(row)

        self._bt_btn = QPushButton("运行回测（当前图表数据）")
        self._bt_btn.clicked.connect(self._on_backtest)
        v.addWidget(self._bt_btn)

        self._bt_summary = QTextBrowser()
        self._bt_summary.setMaximumHeight(160)
        v.addWidget(self._bt_summary)

        self._bt_table = QTableWidget(0, 8)
        self._bt_table.setHorizontalHeaderLabels(
            ["方向", "类型", "入场", "止损", "目标", "出场", "结果", "净R"])
        self._bt_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._bt_table.setMaximumHeight(240)
        v.addWidget(self._bt_table, 1)
        return w

    def _on_backtest(self) -> None:
        if self._frame is None or not getattr(self._frame, "bars", None):
            QMessageBox.information(self, "回测", "当前没有图表数据，请先「获取数据」。")
            return
        from pa_agent.mt5trading.backtest import BacktestParams

        params = BacktestParams(
            lookback=self._sp_lookback.value(),
            stop_buffer_atr=self._sp_stop.value(),
            target_r=self._sp_target.value(),
            cost_points=self._sp_cost.value(),
            enable_limit=self._cb_limit.isChecked(),
            enable_breakout=self._cb_break.isChecked(),
            timeout_bars=int(getattr(self._cfg, "backtest_timeout_bars", 50)) if self._cfg else 50,
        )
        self._bt_btn.setEnabled(False)
        self._status.setText("回测运行中…")
        self._bt_worker = BacktestWorker(self._frame, params, parent=self)
        self._bt_worker.ready.connect(self._on_bt_ready)
        self._bt_worker.failed.connect(self._on_bt_failed)
        self._bt_worker.start()

    def _on_bt_ready(self, result: Any) -> None:
        self._bt_btn.setEnabled(True)
        self._status.setText("回测完成")
        self._bt_summary.setHtml(f"<pre>{result.summary_text()}</pre>")
        self._bt_table.setRowCount(len(result.trades))
        for r, t in enumerate(result.trades):
            vals = [t.side, t.kind, f"{t.entry:g}", f"{t.stop:g}", f"{t.target:g}",
                    f"{t.exit_price:g}", t.outcome, f"{t.r_net:+.2f}"]
            for c, val in enumerate(vals):
                item = QTableWidgetItem(str(val))
                if c == 7:
                    item.setForeground(
                        _qt_color("#D62728" if t.r_net > 0 else "#1A9850"))  # 涨红跌绿
                self._bt_table.setItem(r, c, item)
        self._bt_worker = None

    def _on_bt_failed(self, msg: str) -> None:
        self._bt_btn.setEnabled(True)
        self._status.setText("回测失败")
        self._bt_summary.setHtml(f"<p style='color:#B91C1C'>{msg}</p>")
        self._bt_worker = None

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        w = self._bt_worker
        if w is not None and w.isRunning():
            w.wait(1500)
        super().closeEvent(event)


def _qt_color(hex_color: str):
    from PyQt6.QtGui import QColor

    return QColor(hex_color)
