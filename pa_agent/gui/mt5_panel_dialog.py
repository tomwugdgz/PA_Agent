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
import time
from typing import Any

from PyQt6.QtCore import QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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


class TunerWorker(QThread):
    """后台跑贪婪调参（几十次确定性回测，秒级）。"""

    ready = pyqtSignal(object)    # TuningResult
    failed = pyqtSignal(str)

    def __init__(self, frame: Any, params: Any, parent: Any = None) -> None:
        super().__init__(parent)
        self._frame = frame
        self._params = params

    def run(self) -> None:
        try:
            from pa_agent.journal.greedy_tuner import greedy_tune

            self.ready.emit(greedy_tune(self._frame, self._params))
        except Exception as exc:  # noqa: BLE001
            logger.warning("TunerWorker failed: %s", exc)
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
        self._tune_worker: TunerWorker | None = None
        self._last_tuning: Any = None
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

    # ── Tab 1：下单（参数可编辑，默认按当前 AI 分析填入） ─────────────────────

    def _build_trade_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        self._decision_text = QTextBrowser()
        self._decision_text.setMaximumHeight(140)
        v.addWidget(self._decision_text)
        self._show_decision()

        # ── 可编辑下单参数：AI 分析填默认值，用户可自行修改 ─────────────
        row1 = QHBoxLayout()
        self._cmb_symbol = QComboBox()
        self._cmb_symbol.setEditable(True)
        _meta_sym = str(getattr(getattr(self._record, "meta", None), "symbol", "") or "")
        if _meta_sym:
            self._cmb_symbol.addItem(_meta_sym)
            self._cmb_symbol.setCurrentText(_meta_sym)
        self._cmb_symbol.setToolTip(
            "品种名默认取图表数据源（如 TradingView），与 MT5 经纪商的命名可能不同\n"
            "（例如 XAUUSD vs GOLD、CNHJPY 可能不存在）——下单前请对照\n"
            "MT5「市场报价」窗口确认，名称不同可直接在此修改。")
        self._cb_dir = QComboBox(); self._cb_dir.addItems(["多头", "空头"])
        self._cb_kind = QComboBox(); self._cb_kind.addItems(["市价单", "限价单", "突破单"])
        for label, widget in (("品种", self._cmb_symbol), ("方向", self._cb_dir),
                              ("类型", self._cb_kind)):
            box = QWidget(); hb = QHBoxLayout(box); hb.setContentsMargins(0, 0, 0, 0)
            hb.addWidget(QLabel(label)); hb.addWidget(widget)
            row1.addWidget(box)
        row1.addStretch()
        v.addLayout(row1)

        row2 = QHBoxLayout()

        def _price_spin() -> QDoubleSpinBox:
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 1e9); sp.setDecimals(5); sp.setSingleStep(0.0001)
            return sp

        self._t_entry = _price_spin()
        self._t_stop = _price_spin()
        self._t_tp = _price_spin()
        self._t_lot = QDoubleSpinBox()
        self._t_lot.setRange(0.01, 1000.0); self._t_lot.setDecimals(2)
        self._t_lot.setSingleStep(0.01)
        self._t_lot.setValue(float(getattr(self._cfg, "default_lot", 0.01) or 0.01))
        for label, sp in (("入场", self._t_entry), ("止损", self._t_stop),
                          ("止盈", self._t_tp), ("手数", self._t_lot)):
            box = QWidget(); hb = QHBoxLayout(box); hb.setContentsMargins(0, 0, 0, 0)
            hb.addWidget(QLabel(label)); hb.addWidget(sp)
            row2.addWidget(box)
        v.addLayout(row2)

        hint = QLabel("市价单不需要入场价（留 0 即可）；限价单/突破单必须填入场价；止损必填。")
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        v.addWidget(hint)

        row3 = QHBoxLayout()
        self._fill_btn = QPushButton("① 导入分析")
        self._fill_btn.setToolTip(
            "把主窗口分析结果填进下方参数：优先导入 AI 决策（两阶段分析），\n"
            "没有可执行订单时自动导入最近一次 Laya 报告的价格计划。"
        )
        self._fill_btn.clicked.connect(self._import_analysis)
        self._copy_btn = QPushButton("② 复制下单信息")
        self._copy_btn.setToolTip(
            "把当前下单参数格式化成文本复制到剪贴板，\n"
            "可粘贴到聊天/备忘，或照着在 MT5 手动下单。"
        )
        self._copy_btn.clicked.connect(self._copy_order_info)
        self._connect_btn = QPushButton("连接 MT5")
        self._connect_btn.clicked.connect(self._on_connect)
        self._send_btn = QPushButton("确认后向 MT5 下单")
        self._send_btn.clicked.connect(self._on_send_order)
        if self._cfg is None or not bool(getattr(self._cfg, "enabled", False)):
            self._send_btn.setEnabled(False)
            self._send_btn.setToolTip(
                "已禁用：config/settings.json → mt5trading.enabled 设为 true 后可用"
            )
        for b in (self._fill_btn, self._copy_btn, self._connect_btn, self._send_btn):
            row3.addWidget(b)
        row3.addStretch()
        v.addLayout(row3)
        v.addStretch()
        return w

    def _decision(self) -> dict[str, Any]:
        s2 = getattr(self._record, "stage2_decision", None) if self._record else None
        return s2 if isinstance(s2, dict) else {}

    def _show_decision(self) -> None:
        d = self._decision()
        if not d:
            self._decision_text.setHtml(
                "<p>当前没有 AI 分析决策。可手动设置下方参数直接下单，"
                "或先在主窗口「提交分析」。</p>"
            )
            return
        if str(d.get("order_type") or "") not in ("限价单", "突破单", "市价单"):
            self._decision_text.setHtml(
                "<p>最近一次 AI 分析未给出可执行订单（no_order）。"
                "可手动设置下方参数直接下单。</p>"
            )
            return
        rows = [
            ("方向", d.get("order_direction")), ("订单类型", d.get("order_type")),
            ("入场", d.get("entry_price")), ("止损", d.get("stop_loss_price")),
            ("止盈 TP1", d.get("take_profit_price")),
            ("置信度", d.get("trade_confidence")),
            ("入场依据", d.get("entry_rule")),
        ]
        html = "".join(
            f"<p><b>{k}</b>：{'' if v is None else v}</p>" for k, v in rows
        )
        self._decision_text.setHtml(html)

    def _import_analysis(self) -> None:
        """把主窗口分析结果填进下单参数。

        来源优先级（从高到低）：
        1. **实时图表数据**：从主窗口当前 frame 取最新 K 线 + 指标，
           用确定性规则算出入场/止损/止盈（与 Laya 报告同源但保证最新）
        2. AI 决策（两阶段分析 stage2_decision）——有可执行订单时用
        3. 最近一次 Laya 报告（logs/laya_latest.json）——按报告方向取
           对应价格计划，订单类型默认「限价单」（结构位挂单口径，可改）
        """
        # 优先：用实时图表数据算价格计划
        realtime_plan = self._build_realtime_plan()
        if realtime_plan is not None:
            self._fill_from_realtime_plan(realtime_plan)
            return

        d = self._decision()
        if d and str(d.get("order_type") or "") in ("限价单", "突破单", "市价单"):
            self._fill_from_decision_dict(d, source="AI 决策")
            return

        latest = self._load_latest_laya()
        if latest is not None:
            plan, dire = self._pick_laya_plan(latest)
            if plan is not None:
                self._fill_from_laya_plan(latest, plan, dire)
                return

        QMessageBox.information(
            self, "导入分析",
            "没有可导入的分析结果：\n"
            "· AI 决策为 no_order，且\n"
            "· 本会话尚未生成 Laya 报告（或报告无可执行价格计划）。\n\n"
            "请先在主窗口「提交分析」或生成「Laya 报告」，或手动设置参数。")

    def _build_realtime_plan(self) -> dict[str, Any] | None:
        """从主窗口当前 frame 用确定性规则算价格计划（保证最新）。"""
        try:
            from pa_agent.ai.market_features import compute_simple_market_features
            from pa_agent.report.laya_pricing import plan_long, plan_short
            from pa_agent.util.price_tick import infer_price_tick_from_frame

            frame = self._frame
            if frame is None or not getattr(frame, "bars", None):
                return None

            close = float(frame.bars[0].close)
            atr = None
            indicators = getattr(frame, "indicators", None)
            atr14 = getattr(indicators, "atr14", None) if indicators is not None else None
            if atr14:
                try:
                    atr = float(atr14[0])
                except (TypeError, ValueError):
                    atr = None

            # 如果没有 ATR，无法算价格计划，返回 None 让后续逻辑处理
            if atr is None or atr <= 0:
                return None

            features = compute_simple_market_features(frame)
            tick = infer_price_tick_from_frame(frame)

            # 用 Laya 定价引擎的确定性规则（结构位优先 + ATR 兜底）
            long_p = plan_long(close=close, atr=atr, features=features,
                              cfg=None, tick=tick)
            short_p = plan_short(close=close, atr=atr, features=features,
                                cfg=None, tick=tick)

            # 选可执行的那个计划
            if long_p.actionable:
                return {"direction": "long", "plan": long_p}
            elif short_p.actionable:
                return {"direction": "short", "plan": short_p}
            return None
        except Exception:  # noqa: BLE001
            return None

    def _fill_from_realtime_plan(self, data: dict[str, Any]) -> None:
        """实时图表数据 → 表单。"""
        direction = data["direction"]
        plan = data["plan"]

        self._cb_dir.setCurrentIndex(0 if direction == "long" else 1)
        self._cb_kind.setCurrentIndex(1)  # 限价单：结构位挂单口径
        self._t_entry.setValue(plan.entry if plan.entry else 0.0)
        self._t_stop.setValue(plan.stop if plan.stop else 0.0)
        self._t_tp.setValue(plan.target if plan.target else 0.0)
        self._status.setText(
            f"已导入实时图表数据（{direction}，限价单，可手改）"
        )

    def _fill_from_decision_dict(self, d: dict[str, Any], *, source: str) -> None:
        """AI 决策 → 表单（手数不动）。"""

        def _f(v: Any) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        self._cb_dir.setCurrentIndex(
            0 if str(d.get("order_direction") or "").lower() == "long" else 1)
        kind_map = {"市价单": 0, "限价单": 1, "突破单": 2}
        self._cb_kind.setCurrentIndex(kind_map.get(str(d.get("order_type") or ""), 0))
        self._t_entry.setValue(_f(d.get("entry_price")))
        self._t_stop.setValue(_f(d.get("stop_loss_price")))
        self._t_tp.setValue(_f(d.get("take_profit_price")))
        self._status.setText(f"已导入{source}（可手改）")

    def _load_latest_laya(self) -> dict[str, Any] | None:
        """读取最近一次 Laya 报告（logs/laya_latest.json）。失败返回 None。"""
        try:
            import json as _json

            from pa_agent.config.paths import LOGS_DIR

            path = LOGS_DIR / "laya_latest.json"
            if not path.is_file():
                return None
            return _json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    def _pick_laya_plan(
        self, latest: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str]:
        """按报告方向选对应价格计划；方向缺失时选可执行的那个。

        返回 (plan 字典, 方向 "long"/"short")；无可用计划返回 (None, "")。
        """
        dire = str(latest.get("direction") or "").lower()
        plans = {"long": latest.get("long_plan") or {},
                 "short": latest.get("short_plan") or {}}
        if dire in plans and plans[dire].get("actionable"):
            return plans[dire], dire
        for key in ("long", "short"):
            if plans[key].get("actionable"):
                return plans[key], key
        return None, ""

    def _fill_from_laya_plan(
        self, latest: dict[str, Any], plan: dict[str, Any], dire: str
    ) -> None:
        """Laya 价格计划 → 表单。类型默认限价单（结构位挂单口径）。"""

        def _f(v: Any) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        self._cb_dir.setCurrentIndex(0 if dire == "long" else 1)
        self._cb_kind.setCurrentIndex(1)  # 限价单：结构位挂单，可改
        self._t_entry.setValue(_f(plan.get("entry")))
        self._t_stop.setValue(_f(plan.get("stop")))
        self._t_tp.setValue(_f(plan.get("target")))
        conf = latest.get("direction_confidence")
        conf_txt = f"，方向置信度 {conf:.0%}" if isinstance(conf, (int, float)) else ""
        self._status.setText(
            f"已导入 Laya 报告价格计划（{latest.get('generated_at', '')}{conf_txt}，"
            "类型默认限价单，可改）")

    def _copy_order_info(self) -> None:
        """把当前下单参数格式化成文本复制到剪贴板。"""
        symbol = self._cmb_symbol.currentText().strip() or "未知品种"
        timeframe = str(getattr(getattr(self._record, "meta", None), "timeframe", "") or "")
        direction = self._cb_dir.currentText()
        kind = self._cb_kind.currentText()
        entry = self._t_entry.value()
        stop = self._t_stop.value()
        tp = self._t_tp.value()
        lot = self._t_lot.value()
        lines = [
            f"PA_Agent 下单信息（{time.strftime('%Y-%m-%d %H:%M')}）",
            f"品种：{symbol} {timeframe}",
            f"方向：{direction}　类型：{kind}",
            f"入场：{entry if entry > 0 else '市价'}",
            f"止损：{stop if stop > 0 else '未设'}",
            f"止盈：{tp if tp > 0 else '未设'}",
            f"手数：{lot}",
        ]
        QApplication.clipboard().setText("\n".join(lines))
        self._status.setText("下单信息已复制到剪贴板，可直接粘贴或照着在 MT5 手动下单")

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
        symbol = self._cmb_symbol.currentText().strip()
        if not symbol:
            QMessageBox.information(self, "MT5 下单",
                                    "品种为空——请填入 MT5 经纪商的品种名称"
                                    "（可与图表数据源的命名不同）。")
            return
        direction = "long" if self._cb_dir.currentText() == "多头" else "short"
        kind_map = {"市价单": "market", "限价单": "limit", "突破单": "stop"}
        order_kind = kind_map[self._cb_kind.currentText()]
        entry = self._t_entry.value() if self._t_entry.value() > 0 else None
        stop_loss = self._t_stop.value()
        take_profit = self._t_tp.value() if self._t_tp.value() > 0 else None
        lot = self._t_lot.value()

        if stop_loss <= 0:
            QMessageBox.warning(self, "MT5 下单", "止损价必须大于 0——不带止损的单不允许发出。")
            return
        if order_kind in ("limit", "stop") and not entry:
            QMessageBox.warning(self, "MT5 下单", f"{self._cb_kind.currentText()}必须填入场价。")
            return

        if self._cfg is not None and bool(getattr(self._cfg, "confirm_required", True)):
            ok = QMessageBox.question(
                self, "确认下单",
                "即将向 MT5 发送真实订单：\n\n"
                f"品种：{symbol}　方向：{self._cb_dir.currentText()}　类型：{self._cb_kind.currentText()}\n"
                f"入场：{entry if entry else '市价'}　止损：{stop_loss}\n"
                f"止盈：{take_profit if take_profit else '无'}　手数：{lot}\n\n确认继续？",
            ) == QMessageBox.StandardButton.Yes
            if not ok:
                return
        try:
            from pa_agent.mt5trading.mt5_bridge import OrderRequest, send_order

            req = OrderRequest(
                symbol=symbol,
                direction=direction,
                order_kind=order_kind,
                entry=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot=lot,
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

        # 贪婪调参：AI 在当前图表历史窗口上自动搜索更优参数（只接受回测证据支持的改进）
        tune_row = QHBoxLayout()
        self._tune_btn = QPushButton("贪婪调参（AI 自动优化）")
        self._tune_btn.setToolTip(
            "以回测净收益 R 为目标，对止损缓冲/最小止损/目标R/挂单偏移做贪婪坐标下降；\n"
            "每次评估写入运行记录，找到更优参数后可一键应用。"
        )
        self._tune_btn.clicked.connect(self._on_tune)
        self._apply_btn = QPushButton("应用最优参数到设置")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply_tuned)
        tune_row.addWidget(self._tune_btn)
        tune_row.addWidget(self._apply_btn)
        tune_row.addStretch()
        v.addLayout(tune_row)

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

    def _collect_params(self) -> "Any":
        """从面板控件收集 BacktestParams（回测与调参共用同一口径）。"""
        from pa_agent.mt5trading.backtest import BacktestParams

        return BacktestParams(
            lookback=self._sp_lookback.value(),
            stop_buffer_atr=self._sp_stop.value(),
            target_r=self._sp_target.value(),
            cost_points=self._sp_cost.value(),
            enable_limit=self._cb_limit.isChecked(),
            enable_breakout=self._cb_break.isChecked(),
            timeout_bars=int(getattr(self._cfg, "backtest_timeout_bars", 50)) if self._cfg else 50,
        )

    def _bars_ok(self, need: int) -> bool:
        """回测/调参前预检查 K 线数量，不足时给出明确指引。"""
        n = len(getattr(self._frame, "bars", ()) or ())
        if n >= need:
            return True
        QMessageBox.warning(
            self, "K 线不足",
            f"当前图表 {n} 根，至少需要 {need} 根（预热 60 + 回看窗口）。\n\n"
            "解决办法：主窗口把「K 线根数」调大（推荐 250）后重新「获取数据」。",
        )
        return False

    def _on_backtest(self) -> None:
        if self._frame is None or not getattr(self._frame, "bars", None):
            QMessageBox.information(self, "回测", "当前没有图表数据，请先「获取数据」。")
            return
        if not self._bars_ok(60 + self._sp_lookback.value()):
            return
        params = self._collect_params()
        self._bt_btn.setEnabled(False)
        self._status.setText("回测运行中…")
        self._bt_worker = BacktestWorker(self._frame, params, parent=self)
        self._bt_worker.ready.connect(self._on_bt_ready)
        self._bt_worker.failed.connect(self._on_bt_failed)
        self._bt_worker.start()

    def _on_tune(self) -> None:
        if self._frame is None or not getattr(self._frame, "bars", None):
            QMessageBox.information(self, "贪婪调参", "当前没有图表数据，请先「获取数据」。")
            return
        if not self._bars_ok(60 + self._sp_lookback.value()):
            return
        params = self._collect_params()
        self._tune_btn.setEnabled(False)
        self._bt_btn.setEnabled(False)
        self._apply_btn.setEnabled(False)
        self._status.setText("贪婪调参运行中（几十次回测，约几十秒）…")
        self._tune_worker = TunerWorker(self._frame, params, parent=self)
        self._tune_worker.ready.connect(self._on_tune_ready)
        self._tune_worker.failed.connect(self._on_tune_failed)
        self._tune_worker.start()

    def _on_tune_ready(self, result: Any) -> None:
        self._tune_btn.setEnabled(True)
        self._bt_btn.setEnabled(True)
        self._status.setText("贪婪调参完成")
        self._last_tuning = result
        self._apply_btn.setEnabled(bool(result.improved))
        self._bt_summary.setHtml(f"<pre>{result.summary_text()}</pre>")
        self._tune_worker = None

    def _on_tune_failed(self, msg: str) -> None:
        self._tune_btn.setEnabled(True)
        self._bt_btn.setEnabled(True)
        self._status.setText("贪婪调参失败")
        self._bt_summary.setHtml(f"<p style='color:#B91C1C'>{msg}</p>")
        self._tune_worker = None

    def _on_apply_tuned(self) -> None:
        """把本轮最优参数写回 settings.laya 并持久化到 settings.json。"""
        result = getattr(self, "_last_tuning", None)
        if result is None or not result.improved:
            return
        try:
            from pa_agent.config.paths import SETTINGS_JSON_PATH
            from pa_agent.config.settings import save_settings
            from pa_agent.journal.greedy_tuner import apply_to_settings

            changed = apply_to_settings(self._settings, result)
            save_settings(self._settings, SETTINGS_JSON_PATH)
            QMessageBox.information(
                self, "应用参数",
                "已写入 config/settings.json（重启后仍生效）：\n"
                + "\n".join(changed),
            )
            self._status.setText("；".join(changed))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "应用参数", f"写入设置失败：{exc}")

    def _on_bt_ready(self, result: Any) -> None:
        self._bt_btn.setEnabled(True)
        self._status.setText("回测完成")
        # 回测层记录（旁路，失败静默）
        try:
            from dataclasses import asdict

            from pa_agent.journal.layer_journal import log_backtest

            log_backtest(
                symbol=result.symbol, timeframe=result.timeframe,
                kind="backtest_run", params=asdict(result.params),
                metrics=result.metrics, extra={"role": "manual"},
            )
        except Exception:  # noqa: BLE001
            pass
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
        for w in (self._bt_worker, self._tune_worker):
            if w is not None and w.isRunning():
                w.wait(1500)
        super().closeEvent(event)


def _qt_color(hex_color: str):
    from PyQt6.QtGui import QColor

    return QColor(hex_color)
