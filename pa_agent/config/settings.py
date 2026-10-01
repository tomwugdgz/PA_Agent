"""Pydantic settings models for PA Agent."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DecisionStance = Literal["conservative", "balanced", "aggressive", "extreme_aggressive"]
DataSourceKind = Literal["mt5", "tradingview", "akshare", "eastmoney", "eastmoney_futures", "tushare"]
NormalizationMode = Literal["strict", "lenient"]


class AIProviderSettings(BaseModel):
    """AI provider connection and behaviour settings."""
    model_config = ConfigDict(extra="ignore")

    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    api_key_encrypted: str = ""
    thinking: bool = True
    reasoning_effort: Literal["low", "medium", "high", "max"] = "high"
    context_window: int = 2_000_000


class PromptSettings(BaseModel):
    """Prompt assembly tuning (accuracy-oriented defaults)."""
    model_config = ConfigDict(extra="ignore")

    #: When True, Stage 2 loads every strategy .txt (legacy/test behaviour).
    stage2_load_full_strategy_library: bool = False
    experience_max_entries: int = Field(default=0, ge=0, le=10)
    experience_max_chars_per_entry: int = Field(default=400, ge=100, le=4000)
    #: Inject pattern判定表 + 速查 brief into Stage 1 user prompt (reduces missed tags).
    stage1_inject_pattern_briefs: bool = True


class ValidationSettings(BaseModel):
    """Post-LLM validation behaviour."""
    model_config = ConfigDict(extra="ignore")

    normalization_mode: NormalizationMode = "lenient"
    #: Stage-1 cross-field checks (gate trace, bar_by_bar, pattern tags). Off by default.
    stage1_coherence_checks: bool = False
    #: Stage-2 trace / diagnosis cross-checks (not order safety). Off by default.
    stage2_coherence_checks: bool = False
    trace_semantic_checks: bool = False
    strict_bar_by_bar_features: bool = False
    #: Allow Stage 1 truncated JSON tail repair before failing syntax validation.
    disable_truncation_repair: bool = False
    #: Re-call API with structured feedback when validation fails (format errors).
    retry_enabled: bool = True
    retry_max: int = Field(default=3, ge=0, le=5)
    #: Max retries for category=c semantic errors (subset only).
    retry_max_semantic: int = Field(default=1, ge=0, le=3)
    retry_stage2: bool = True


class GeneralSettings(BaseModel):
    """UI and data-feed general settings."""
    model_config = ConfigDict(extra="ignore")

    analysis_bar_count: int = Field(default=100, ge=2, le=5000)
    refresh_interval_ms: int = 1000
    context_warning_threshold_pct: float = 99_999_999.0
    last_data_source: DataSourceKind = "mt5"
    #: A-share K-line adjust for East Money / Baostock (qfq=前复权)
    kline_adjust: Literal["qfq", "hfq", "none"] = "qfq"
    #: TradingView 交易所；空字符串 =（自动）依次探测预设列表
    last_tradingview_exchange: str = ""
    last_symbol: str = "XAUUSDm"
    last_timeframe: str = "15m"
    decision_flow_auto_play: bool = True
    decision_flow_play_seconds: int = 50
    #: 阶段二给出限价/突破/市价单时：警报音、弹窗，并自动切到「决策」页（跳过决策树可视化演示）
    alert_on_order_opportunity: bool = True
    incremental_max_new_bars: int = Field(default=10, ge=0, le=500)
    #: 阶段二交易倾向：balanced=默认；conservative/aggressive 逐级调整下单意愿
    decision_stance: DecisionStance = "balanced"
    #: 决策树可视化：在「整图适配」基础上的缩放百分比（100=与适配一致；可任意放大，仅下限 10%）
    decision_flow_default_zoom_pct: int = Field(default=600, ge=10)
    #: 「实时」页思考过程/撰写回答框与追问输入框的等宽字体字号（pt）
    stream_pane_font_pt: int = Field(default=11, ge=8, le=28)
    #: K 线图上 #序号 标签的字号（pt）
    chart_seq_label_font_pt: int = Field(default=11, ge=6, le=24)
    #: 两阶段分析结束后是否自动恢复 K 线图表实时刷新
    auto_resume_chart_after_analysis: bool = False
    #: 持续跟踪分析：有新K线收盘时自动触发新一轮分析
    keep_analysis: bool = False
    #: 重试后取消持续跟踪分析：校验失败触发重试后自动关闭 keep_analysis
    cancel_keep_analysis_on_retry: bool = False
    #: 交易决策置信度门槛：仅当 trade_confidence >= 此值时，才视为有下单机会（弹窗警报并提供决策详情）
    decision_confidence_threshold: int = Field(default=40, ge=0, le=100)
    #: 开启下根K线预期功能；关闭时不向模型请求该预测，节省 token
    enable_next_bar_prediction: bool = False
    #: 同一结构位 entry 相差≤3跳时，禁止反向新方案的冷却 K 线根数（已收盘）
    structure_flip_cooldown_bars: int = Field(default=3, ge=1, le=50)

    @field_validator("last_data_source", mode="before")
    @classmethod
    def _coerce_legacy_data_source(cls, v: object) -> object:
        if v == "yfinance":
            return "eastmoney"
        if v in ("adata", "a_share"):
            return "akshare"
        if v == "eastmoney":
            return "eastmoney"
        if v == "tushare":
            return "tushare"
        return v

    @field_validator("decision_flow_default_zoom_pct", mode="before")
    @classmethod
    def _coerce_zoom_pct(cls, v: object) -> object:
        if v is None:
            return 50
        return v


_FEISHU_CONFIG_KEYS = (
    "enabled",
    "webhook_url",
    "secret",
    "app_id",
    "app_secret",
    "notify_on_order_only",
)


class FeishuSettings(BaseModel):
    """Feishu bot notification settings (persisted in settings.json)."""
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    webhook_url: str = ""
    secret: str = ""
    app_id: str = ""
    app_secret: str = ""
    #: True = only push when there is an order opportunity.
    notify_on_order_only: bool = True


class TushareSettings(BaseModel):
    """Tushare Pro data source settings (persisted in ignored settings.json)."""
    model_config = ConfigDict(extra="ignore")

    token: str = ""


class PushPlusSettings(BaseModel):
    """PushPlus notification settings (settings.json only; no GUI)."""
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    token: str = ""


class LayaSettings(BaseModel):
    """Laya System-1 决策模型设置（本地离线推理，与 DeepSeek 两阶段分析解耦）。

    Laya 是非自回归判别模型，只输出 choice / score / noul 三种结构化原语，
    不产出自然语言文本，因此：
      - 「分析报告」正文由本地代码渲染；
      - 「买入/卖出价格」由确定性结构位 + ATR 计算；
      -     Laya 只负责提供方向/结构/信号有效性的**概率**。
    """
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    #: 总开关。关闭时工具栏「Laya 报告」按钮置灰。
    enabled: bool = True
    #: 权重根目录（内含 multilingual/ 子目录）。Agent 检测到本地目录存在即跳过联网下载。
    #: 默认放在当前用户主目录 ~/laya-models/laya（Windows 即
    #: C:\Users\<用户名>\laya-models\laya），可在 config/settings.json 覆盖。
    model_dir: str = str(Path.home() / "laya-models" / "laya")
    #: 权重子目录：multilingual 支持中文（含 CJK 路由），"" 为英文档
    subfolder: str = "multilingual"
    #: 推理设备：auto=有 CUDA 就用 GPU，否则 CPU
    device: str = "auto"
    #: 低置信度门槛：低于此值的答案在报告中标注「不可信」，不参与价格建议生成
    min_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    #: 是否把每次 Laya 调用写入标注数据集（用于后续微调），见 ai/laya_annotation.py
    collect_annotations: bool = True

    #: ── 定价参数（混合口径：结构优先 + ATR 兜底） ─────────────────────────
    #: 挂单相对结构位的偏移倍数 × ATR（做多挂 support + k*ATR）
    entry_offset_atr: float = Field(default=0.10, ge=0.0, le=2.0)
    #: 止损相对结构失效位的缓冲倍数 × ATR
    stop_buffer_atr: float = Field(default=0.25, ge=0.0, le=3.0)
    #: 无 measured_move 时的兜底目标：R 倍数（R = |entry - stop|）
    fallback_target_r: float = Field(default=2.0, ge=0.5, le=10.0)
    #: 无结构位时的纯 ATR 兜底止损倍数
    fallback_stop_atr: float = Field(default=1.0, ge=0.1, le=5.0)


class MT5Settings(BaseModel):
    """MT5 自动交易设置（Python 直连 order_send）。

    安全设计：下单**必须**经 GUI 确认弹窗，无任何自动循环下单路径；
    enabled 默认 False，需用户显式开启。
    """
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    #: 总开关：False 时 MT5 面板只读（连接/账户信息可用，下单按钮禁用）
    enabled: bool = False
    #: EA/订单魔号：识别本程序所下订单，便于平仓与统计
    magic: int = Field(default=20260930, ge=1)
    #: 默认下单手数
    default_lot: float = Field(default=0.01, gt=0.0)
    #: 最大允许点差（point 为单位）；实际点差超过此值拒绝下单，0=不检查
    max_spread_points: int = Field(default=0, ge=0)
    #: 下单前强制确认弹窗（建议保持 True；关闭属高风险行为）
    confirm_required: bool = True
    #: 挂单有效期（根 K 线）；到期未成交自动撤单，0=不过期（由 EA 端 EXP_DOU 条件控制）
    pending_expiry_bars: int = Field(default=12, ge=0, le=500)
    #: MT5 终端路径（留空 = 自动附加到已登录的运行中终端）
    terminal_path: str = ""
    #: 回测参数：信号回看窗口（根）
    backtest_lookback: int = Field(default=100, ge=20, le=2000)
    #: 回测：超时平仓根数（TP/SL 都没触发时按收盘价离场）
    backtest_timeout_bars: int = Field(default=50, ge=5, le=1000)
    #: 回测：单边成本（点/笔，模拟点差+滑点），从盈利中扣除
    backtest_cost_points: int = Field(default=10, ge=0)


class Settings(BaseModel):
    """Root settings object persisted to config/settings.json."""
    model_config = ConfigDict(extra="ignore")

    provider: AIProviderSettings = Field(default_factory=AIProviderSettings)
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    prompt: PromptSettings = Field(default_factory=PromptSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    feishu: FeishuSettings = Field(default_factory=FeishuSettings)
    pushplus: PushPlusSettings = Field(default_factory=PushPlusSettings)
    tushare: TushareSettings = Field(default_factory=TushareSettings)
    laya: LayaSettings = Field(default_factory=LayaSettings)
    mt5trading: MT5Settings = Field(default_factory=MT5Settings)


def provider_api_key_configured(settings: Settings | None) -> bool:
    """Return True when a non-empty API key is loaded in memory."""
    if settings is None:
        return False
    return bool((settings.provider.api_key or "").strip())


# ── Persistence ───────────────────────────────────────────────────────────────
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _migrate_legacy_feishu_json(raw: dict, settings_path: Path) -> bool:
    """Merge legacy config/feishu.json into settings.feishu when needed."""
    legacy_path = settings_path.parent / "feishu.json"
    if not legacy_path.exists():
        return False

    feishu = raw.setdefault("feishu", {})
    if (feishu.get("webhook_url") or "").strip():
        return False

    try:
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("legacy feishu.json unreadable (%s); skipping migration", exc)
        return False

    migrated = False
    for key in _FEISHU_CONFIG_KEYS:
        if key not in legacy:
            continue
        value = legacy.get(key)
        if value in (None, ""):
            continue
        if feishu.get(key) in (None, ""):
            feishu[key] = value
            migrated = True
    if migrated:
        logger.info("Migrated Feishu config from %s into settings.json", legacy_path)
    return migrated


def load_settings(path: Path | None = None) -> "Settings":
    """Load settings from *path* (default: SETTINGS_JSON_PATH).

    Returns default Settings and writes them to disk if the file is absent.
    """
    from pa_agent.config.paths import SETTINGS_JSON_PATH

    path = path or SETTINGS_JSON_PATH

    if not path.exists():
        defaults = Settings()
        save_settings(defaults, path)
        return defaults

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("settings.json unreadable (%s); using defaults", exc)
        return Settings()

    # Migrate legacy field names
    general = raw.get("general", {})
    if "cost_warning_threshold_pct" in general and "context_warning_threshold_pct" not in general:
        general["context_warning_threshold_pct"] = general.pop("cost_warning_threshold_pct")
    general.pop("last_htf_text", None)
    from pa_agent.data.market_defaults import migrate_general_gold_defaults

    migrate_general_gold_defaults(general)
    if "default_bar_count" in general and "analysis_bar_count" not in general:
        general["analysis_bar_count"] = general.pop("default_bar_count")
    raw["general"] = general
    provider = raw.get("provider", {})
    provider.pop("pricing", None)
    raw["provider"] = provider

    # Migrate legacy encrypted key: drop it, api_key already in provider dict
    raw.setdefault("provider", {}).setdefault("api_key", "")

    migrated_feishu = _migrate_legacy_feishu_json(raw, path)
    settings = Settings.model_validate(raw)
    dirty = migrated_feishu
    if settings.pushplus.enabled and not settings.pushplus.token.strip():
        if not (os.environ.get("PUSHPLUS_TOKEN") or "").strip():
            settings.pushplus.enabled = False
            logger.info(
                "PushPlus enabled but token empty — auto-disabled "
                "(Feishu notifications unaffected)"
            )
            dirty = True
    if dirty:
        save_settings(settings, path)
    return settings


def save_settings(settings: "Settings", path: Path | None = None) -> None:
    """Persist settings to *path* (default: SETTINGS_JSON_PATH)."""
    from pa_agent.config.paths import SETTINGS_JSON_PATH

    path = path or SETTINGS_JSON_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    data = settings.model_dump()

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
