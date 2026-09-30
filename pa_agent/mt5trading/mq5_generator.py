# -*- coding: utf-8 -*-
"""MQL5 EA 生成器：把 PA_Agent 的决策/规则落成 .mq5 源码。

两份模板
--------
1. **决策执行 EA**（`generate_decision_ea`）：把当前 AI 决策
   （方向/订单类型/入场/止损/止盈/手数）写成 EA 的输入参数默认值，
   挂到图表即自动执行这一单。适合「AI 判断完 → 一键拿到可执行的 MT5 程序」。
2. **策略回测 EA**（`generate_strategy_ea`）：完整确定性策略
   （回踩限价 + 突破市价、结构失效位止损、R 倍数目标、超时平仓、铁丝网过滤），
   与 Python 回测引擎同一口径，供 MT5 策略测试器回测与实盘。

注意：MQL5 代码在本机无法编译验证（需 MetaEditor），模板刻意只用最常规的
CTrade / iATR / 摆动分形写法，降低编译失败概率。首次使用请先在策略测试器跑通。
"""
from __future__ import annotations

import time
from typing import Any


def _fmt(v: float | None, default: float = 0.0) -> str:
    if v is None:
        return f"{default}"
    return f"{float(v):.10f}".rstrip("0").rstrip(".")


def _esc(text: str) -> str:
    return (text or "").replace("\\", "\\\\").replace('"', '\\"')


# ═════════════════════════════════════════════════════════════════════════════
# 模板一：决策执行 EA
# ═════════════════════════════════════════════════════════════════════════════

_DECISION_EA = """//+------------------------------------------------------------------+
//| PA_Agent 决策执行 EA（自动生成，勿手改参数区外的逻辑）            |
//| 来源：PA_Agent AI 两阶段分析                                      |
//| 生成时间：{generated_at}                                  |
//+------------------------------------------------------------------+
//| AI 决策快照                                                       |
//|  品种/周期 : {symbol} {timeframe}                          |
//|  方向      : {direction_zh}                                       |
//|  订单类型  : {order_kind_zh}                                      |
//|  入场      : {entry}                                              |
//|  止损      : {stop}                                               |
//|  止盈 TP1  : {tp1}                                               |
//|  止盈 TP2  : {tp2}（MT5 单笔只支持一个 TP，已写入注释供手动分批）|
//|  置信度    : {trade_confidence}%                                  |
//|  入场依据  : {entry_rule}                                        |
//|  失效条件  : {invalidation}                                      |
//+------------------------------------------------------------------+
#property strict
#include <Trade\\Trade.mqh>

//── 参数区（默认值=AI 决策，可改后重新挂载） ──────────────────────────
input int    InpDirection    = {direction_code};   // 0=买入 1=卖出
input int    InpOrderKind    = {kind_code};        // 0=市价 1=限价 2=突破
input double InpEntry        = {entry};            // 入场价（市价单忽略）
input double InpStopLoss     = {stop};             // 止损
input double InpTakeProfit   = {tp1};              // 止盈 TP1
input double InpLots         = {lot};              // 手数
input int    InpExpiryBars   = {expiry_bars};      // 挂单有效期（根K线，0=不过期）
input int    InpMagic        = {magic};            // 魔号
input int    InpMaxSpread    = {max_spread};       // 最大点差(point)，0=不检查

CTrade   trade;
bool     g_done = false;          // 只执行一次
datetime g_placed = 0;            // 挂单时间（用于到期撤单）

int OnInit()
{{
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(20);

   // 基本校验
   if(InpStopLoss <= 0)                {{ Print("参数错误：止损必须>0");      return INIT_PARAMETERS_INCORRECT; }}
   if(InpLots <= 0)                    {{ Print("参数错误：手数必须>0");      return INIT_PARAMETERS_INCORRECT; }}
   if(InpOrderKind != 0 && InpEntry <= 0) {{ Print("参数错误：挂单需有效入场价"); return INIT_PARAMETERS_INCORRECT; }}
   if(InpDirection == 0 && InpStopLoss >= (InpOrderKind == 0 ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : InpEntry))
      {{ Print("参数错误：多头止损必须低于入场"); return INIT_PARAMETERS_INCORRECT; }}
   if(InpDirection == 1 && InpStopLoss <= (InpOrderKind == 0 ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : InpEntry))
      {{ Print("参数错误：空头止损必须高于入场"); return INIT_PARAMETERS_INCORRECT; }}

   if(InpMaxSpread > 0)
   {{
      long spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
      if(spread > InpMaxSpread) {{ Print("点差 ", spread, " 超上限 ", InpMaxSpread, "，放弃"); return INIT_PARAMETERS_INCORRECT; }}
   }}

   Print("PA_Agent 决策执行 EA 就绪：", InpDirection == 0 ? "买入" : "卖出",
         " kind=", InpOrderKind, " entry=", InpEntry, " sl=", InpStopLoss, " tp=", InpTakeProfit);
   return INIT_SUCCEEDED;
}}

void OnTick()
{{
   if(g_done) return;

   bool ok = false;
   if(InpOrderKind == 0)                    // 市价
   {{
      ok = (InpDirection == 0)
         ? trade.Buy(InpLots, _Symbol, 0, InpStopLoss, InpTakeProfit, "PA_Agent")
         : trade.Sell(InpLots, _Symbol, 0, InpStopLoss, InpTakeProfit, "PA_Agent");
   }}
   else                                     // 挂单
   {{
      datetime expiry = 0;
      if(InpExpiryBars > 0)
         expiry = TimeCurrent() + (datetime)InpExpiryBars * PeriodSeconds();
      trade.SetTypeFillingBySymbol(_Symbol);
      if(InpDirection == 0)
         ok = (InpOrderKind == 1)
            ? trade.BuyLimit(InpLots, InpEntry, _Symbol, InpStopLoss, InpTakeProfit,
                             ORDER_TIME_SPECIFIED, expiry, "PA_Agent")
            : trade.BuyStop(InpLots, InpEntry, _Symbol, InpStopLoss, InpTakeProfit,
                            ORDER_TIME_SPECIFIED, expiry, "PA_Agent");
      else
         ok = (InpOrderKind == 1)
            ? trade.SellLimit(InpLots, InpEntry, _Symbol, InpStopLoss, InpTakeProfit,
                              ORDER_TIME_SPECIFIED, expiry, "PA_Agent")
            : trade.SellStop(InpLots, InpEntry, _Symbol, InpStopLoss, InpTakeProfit,
                             ORDER_TIME_SPECIFIED, expiry, "PA_Agent");
   }}

   if(ok)
   {{
      g_done = true;
      Print("PA_Agent 订单已提交：ticket=", trade.ResultOrder());
   }}
   else
   {{
      Print("下单失败 retcode=", trade.ResultRetcode(), " comment=", trade.ResultComment());
      // 连续失败不自动重试（避免风暴）；如需重试请人工重新挂载 EA
      g_done = true;
   }}
}}

// 挂单到期撤单兜底（broker 不支持 ORDER_TIME_SPECIFIED 时生效）
void OnTimer(){{}}
//+------------------------------------------------------------------+
"""


def generate_decision_ea(
    *,
    decision: dict[str, Any],
    symbol: str,
    timeframe: str,
    magic: int,
    lot: float,
    max_spread_points: int,
    expiry_bars: int,
) -> str:
    """把 stage2 决策 dict 渲染成可编译的 .mq5 源码。

    decision 缺字段时用保守默认（no_order 会抛 ValueError——无单不生成）。
    """
    direction = str(decision.get("order_direction") or "").lower()
    order_kind = str(decision.get("order_type") or "").strip()
    kind_map = {"市价单": 0, "限价单": 1, "突破单": 2}
    if order_kind not in kind_map:
        raise ValueError(f"无法生成 EA：订单类型 {order_kind!r} 不在（市价单/限价单/突破单）")
    dir_code = 0 if direction in ("long", "buy", "多", "买入") else (
        1 if direction in ("short", "sell", "空", "卖出") else -1)
    if dir_code < 0:
        raise ValueError(f"无法生成 EA：方向 {direction!r} 无法识别")

    entry = decision.get("entry_price")
    stop = decision.get("stop_loss_price")
    tp1 = decision.get("take_profit_price")
    if stop is None or float(stop) <= 0:
        raise ValueError("无法生成 EA：决策缺少止损价")
    if order_kind != "市价单" and (entry is None or float(entry) <= 0):
        raise ValueError("无法生成 EA：挂单缺少入场价")

    s1 = str(decision.get("reasoning") or "")
    s2 = str(decision.get("invalidation_condition") or "")
    rule = str(decision.get("entry_rule") or "")
    conf = decision.get("trade_confidence")
    conf_s = str(conf) if conf is not None else "—"

    # AI 决策理由写进文件头注释块（MQL5 单行注释，防换行破坏结构）
    return _DECISION_EA.format(
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        symbol=_esc(symbol), timeframe=_esc(timeframe),
        direction_zh="买入/多头" if dir_code == 0 else "卖出/空头",
        order_kind_zh=order_kind,
        entry=_fmt(entry), stop=_fmt(stop), tp1=_fmt(tp1),
        tp2=_fmt(decision.get("take_profit_price_2")),
        trade_confidence=_esc(conf_s),
        entry_rule=_esc(rule), invalidation=_esc(s2),
        direction_code=dir_code, kind_code=kind_map[order_kind],
        lot=_fmt(lot), expiry_bars=expiry_bars, magic=magic,
        max_spread=max_spread_points,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 模板二：策略回测 EA（与 Python 回测引擎同口径）
# ═════════════════════════════════════════════════════════════════════════════

_STRATEGY_EA = """//+------------------------------------------------------------------+
//| PA_Agent 策略 EA（自动生成）—— 结构位+ATR 确定性规则             |
//| 口径：回踩限价(支撑/阻力) + 收盘突破市价；失效位止损；R 倍数目标 |
//| 与 PA_Agent 内置回测引擎同源，供 MT5 策略测试器验证与实盘        |
//| 生成时间：{generated_at}                                  |
//+------------------------------------------------------------------+
#property strict
#include <Trade\\Trade.mqh>

//── 策略参数 ──────────────────────────────────────────────────────────
input int    InpLookback        = {lookback};    // 结构回看窗口（根）
input int    InpATRPeriod       = 14;            // ATR 周期
input double InpEntryOffsetATR  = {entry_offset};  // 挂单相对结构位的 ATR 偏移
input double InpStopBufferATR   = {stop_buffer};   // 止损相对失效位的 ATR 缓冲
input double InpMinStopATR      = {min_stop};      // 最小止损距离（×ATR，盖过噪音）
input double InpTargetR         = {target_r};      // 目标（R 倍数，无 MM 投影时）
input int    InpTouchWindow     = {touch_window};  // 近端测试窗口（根）
input bool   InpEnableLimit     = true;          // 启用限价单（回踩）
input bool   InpEnableBreakout  = true;          // 启用突破单
input int    InpPendingExpiry   = {pending_expiry}; // 挂单有效期（根）
input int    InpTimeoutBars     = {timeout_bars};  // 持仓超时（根，按收盘平仓）
input double InpFixedLots       = {lot};           // 固定手数
input int    InpMagic           = {magic};         // 魔号
input int    InpMaxSpread       = {max_spread};    // 最大点差(point)，0=不检查
input int    InpCooldownBars    = 1;             // 平仓后冷却（根）

//── 全局 ──────────────────────────────────────────────────────────────
CTrade   trade;
int      g_atr = INVALID_HANDLE;
datetime g_lastBar = 0;
datetime g_pendingPlaced = 0;
datetime g_cooldownUntil = 0;

int OnInit()
{{
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(20);
   g_atr = iATR(_Symbol, PERIOD_CURRENT, InpATRPeriod);
   if(g_atr == INVALID_HANDLE) {{ Print("iATR 句柄创建失败"); return INIT_FAILED; }}
   EventSetTimer(30);
   return INIT_SUCCEEDED;
}}

void OnDeinit(const int reason) {{ EventKillTimer(); IndicatorRelease(g_atr); }}

double Atr(int shift)
{{
   double buf[1];
   if(CopyBuffer(g_atr, 0, shift, 1, buf) != 1) return 0.0;
   return buf[0];
}}

// 摆动点：左右各 2 根确认
bool IsSwingHigh(int shift)
{{
   double h = iHigh(_Symbol, PERIOD_CURRENT, shift);
   for(int k = 1; k <= 2; k++)
   {{
      if(iHigh(_Symbol, PERIOD_CURRENT, shift - k) >= h) return false;
      if(iHigh(_Symbol, PERIOD_CURRENT, shift + k) >= h) return false;
   }}
   return true;
}}

bool IsSwingLow(int shift)
{{
   double l = iLow(_Symbol, PERIOD_CURRENT, shift);
   for(int k = 1; k <= 2; k++)
   {{
      if(iLow(_Symbol, PERIOD_CURRENT, shift - k) <= l) return false;
      if(iLow(_Symbol, PERIOD_CURRENT, shift + k) <= l) return false;
   }}
   return true;
}}

// 最近结构位：支撑=低于现价的最近摆动低点；阻力=高于现价的最近摆动高点
void FindLevels(double close, double &support, double &resistance)
{{
   support = 0; resistance = 0;
   double bestSup = 0, bestRes = 0;
   for(int s = 3; s <= InpLookback; s++)
   {{
      if(IsSwingLow(s))
      {{
         double v = iLow(_Symbol, PERIOD_CURRENT, s);
         if(v < close && v > bestSup) bestSup = v;
      }}
      if(IsSwingHigh(s))
      {{
         double v = iHigh(_Symbol, PERIOD_CURRENT, s);
         if(v > close && (bestRes == 0 || v < bestRes)) bestRes = v;
      }}
   }}
   support = bestSup; resistance = bestRes;
}}

// 铁丝网近似：近 10 根重叠率过高 → 不做
bool Barbwire()
{{
   int overl = 0;
   for(int s = 1; s <= 10; s++)
   {{
      double h0 = iHigh(_Symbol, PERIOD_CURRENT, s), l0 = iLow(_Symbol, PERIOD_CURRENT, s);
      double h1 = iHigh(_Symbol, PERIOD_CURRENT, s + 1), l1 = iLow(_Symbol, PERIOD_CURRENT, s + 1);
      double rng = h1 - l1;
      if(rng > 0 && (MathMin(h0, h1) - MathMax(l0, l1)) / rng > 0.5) overl++;
   }}
   return overl >= 7;   // 10 根里 7 根高度重叠
}}

bool HasOurPosition()
{{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {{
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0 && PositionSelectByTicket(ticket)
         && PositionGetInteger(POSITION_MAGIC) == InpMagic
         && PositionGetString(POSITION_SYMBOL) == _Symbol) return true;
   }}
   return false;
}}

bool HasOurPending()
{{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {{
      ulong ticket = OrderGetTicket(i);
      if(ticket > 0 && OrderGetInteger(ORDER_MAGIC) == InpMagic
         && OrderGetString(ORDER_SYMBOL) == _Symbol) return true;
   }}
   return false;
}}

void CancelOurPending()
{{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {{
      ulong ticket = OrderGetTicket(i);
      if(ticket > 0 && OrderGetInteger(ORDER_MAGIC) == InpMagic
         && OrderGetString(ORDER_SYMBOL) == _Symbol)
         trade.OrderDelete(ticket);
   }}
   g_pendingPlaced = 0;
}}

// 超时平仓：持仓时间超过 InpTimeoutBars 根
void CheckTimeout()
{{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {{
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0 || !PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      datetime opened = (datetime)PositionGetInteger(POSITION_TIME);
      if(TimeCurrent() - opened >= (long)InpTimeoutBars * PeriodSeconds())
      {{
         trade.PositionClose(ticket);
         g_cooldownUntil = TimeCurrent() + InpCooldownBars * PeriodSeconds();
      }}
   }}
}}

void OnTick()
{{
   // 新根判定：只在收盘后的第一跳执行一次
   datetime barTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(barTime == g_lastBar) {{ CheckTimeout(); return; }}
   g_lastBar = barTime;
   CheckTimeout();

   if(TimeCurrent() < g_cooldownUntil) return;
   if(HasOurPosition() || HasOurPending()) return;

   if(InpMaxSpread > 0)
   {{
      long spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
      if(spread > InpMaxSpread) return;
   }}

   double atr = Atr(1);                       // 用刚收盘那根的 ATR
   if(atr <= 0) return;
   if(Barbwire()) return;                     // 铁丝网：不做

   double close1 = iClose(_Symbol, PERIOD_CURRENT, 1);
   double support = 0, resistance = 0;
   FindLevels(close1, support, resistance);

   // 近端测试窗口：最近 InpTouchWindow 根内触碰过结构位
   double minLow = DBL_MAX, maxHigh = 0;
   for(int s = 1; s <= InpTouchWindow; s++)
   {{
      minLow  = MathMin(minLow,  iLow(_Symbol, PERIOD_CURRENT, s));
      maxHigh = MathMax(maxHigh, iHigh(_Symbol, PERIOD_CURRENT, s));
   }}

   double lots = InpFixedLots;

   //── 多头：回踩支撑挂限价多
   if(InpEnableLimit && support > 0)
   {{
      bool tested   = (minLow <= support + 0.5 * atr);
      bool holding  = (close1 > support);
      if(tested && holding)
      {{
         double entry = NormalizeDouble(support + InpEntryOffsetATR * atr, _Digits);
         double sl    = NormalizeDouble(support - MathMax(InpStopBufferATR, InpMinStopATR) * atr, _Digits);
         double tp    = NormalizeDouble(entry + InpTargetR * (entry - sl), _Digits);
         if(sl < entry && entry < tp)
         {{
            if(trade.BuyLimit(lots, entry, _Symbol, sl, tp, ORDER_TIME_GTC, 0, "PA_Agent"))
            {{
               g_pendingPlaced = TimeCurrent();
               Print("PA_Agent 限价多挂单 entry=", entry, " sl=", sl, " tp=", tp);
               return;
            }}
         }}
      }}
   }}

   //── 空头：反抽阻力挂限价空
   if(InpEnableLimit && resistance > 0)
   {{
      bool tested   = (maxHigh >= resistance - 0.5 * atr);
      bool holding  = (close1 < resistance);
      if(tested && holding)
      {{
         double entry = NormalizeDouble(resistance - InpEntryOffsetATR * atr, _Digits);
         double sl    = NormalizeDouble(resistance + MathMax(InpStopBufferATR, InpMinStopATR) * atr, _Digits);
         double tp    = NormalizeDouble(entry - InpTargetR * (sl - entry), _Digits);
         if(tp < entry && entry < sl)
         {{
            if(trade.SellLimit(lots, entry, _Symbol, sl, tp, ORDER_TIME_GTC, 0, "PA_Agent"))
            {{
               g_pendingPlaced = TimeCurrent();
               Print("PA_Agent 限价空挂单 entry=", entry, " sl=", sl, " tp=", tp);
               return;
            }}
         }}
      }}
   }}

   //── 突破确认市价（收盘越过结构位，且上一根仍在其内）
   double close2 = iClose(_Symbol, PERIOD_CURRENT, 2);
   if(InpEnableBreakout && resistance > 0 && close1 > resistance && close2 <= resistance)
   {{
      double entry = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl    = NormalizeDouble(MathMin(resistance - MathMax(InpStopBufferATR, InpMinStopATR) * atr, entry - 0.5 * atr), _Digits);
      double tp    = NormalizeDouble(entry + InpTargetR * (entry - sl), _Digits);
      if(sl < entry && entry < tp)
      {{
         if(trade.Buy(lots, _Symbol, 0, sl, tp, "PA_Agent-brk"))
         {{
            g_cooldownUntil = 0;
            Print("PA_Agent 突破市价多 entry=", entry, " sl=", sl, " tp=", tp);
            return;
         }}
      }}
   }}
   if(InpEnableBreakout && support > 0 && close1 < support && close2 >= support)
   {{
      double entry = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl    = NormalizeDouble(MathMax(support + MathMax(InpStopBufferATR, InpMinStopATR) * atr, entry + 0.5 * atr), _Digits);
      double tp    = NormalizeDouble(entry - InpTargetR * (sl - entry), _Digits);
      if(tp < entry && entry < sl)
      {{
         if(trade.Sell(lots, _Symbol, 0, sl, tp, "PA_Agent-brk"))
         {{
            Print("PA_Agent 突破市价空 entry=", entry, " sl=", sl, " tp=", tp);
            return;
         }}
      }}
   }}

   //── 挂单超期撤单
   if(g_pendingPlaced > 0
      && TimeCurrent() - g_pendingPlaced >= (long)InpPendingExpiry * PeriodSeconds())
      CancelOurPending();
}}

void OnTimer() {{ CheckTimeout(); }}
//+------------------------------------------------------------------
"""


def generate_strategy_ea(
    *,
    magic: int,
    lot: float,
    max_spread_points: int,
    lookback: int = 100,
    entry_offset_atr: float = 0.10,
    stop_buffer_atr: float = 0.25,
    min_stop_atr: float = 1.0,
    target_r: float = 2.0,
    touch_window: int = 3,
    pending_expiry: int = 12,
    timeout_bars: int = 50,
) -> str:
    """渲染策略回测 EA。参数与 Python 回测引擎一一对应。"""
    return _STRATEGY_EA.format(
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        lookback=lookback, entry_offset=_fmt(entry_offset_atr),
        stop_buffer=_fmt(stop_buffer_atr), min_stop=_fmt(min_stop_atr), target_r=_fmt(target_r),
        touch_window=touch_window, pending_expiry=pending_expiry,
        timeout_bars=timeout_bars, lot=_fmt(lot), magic=magic,
        max_spread=max_spread_points,
    )


def save_mq5(source: str, base_dir: Any, name: str) -> Any:
    """把 .mq5 写入 base_dir（默认 logs/mql5/），返回实际路径。"""
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.mq5"
    path.write_text(source, encoding="utf-8")
    return path
