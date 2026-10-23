"""
分布式 VectorBT 策略参数网格回测 Worker (Distributed Backtest Worker)
参考:
- Ray Plasma Zero-Copy Object Store Architecture
- VectorBT Pro Distributed Grid Search Pattern
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import pyarrow as pa

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

# 优先探测增强型 VectorBT 引擎
try:
    from engines.vectorbt_adapter.ashare_sim import EnhancedVectorBTEngine
    from contracts import ExecutionAssumptions, ExecutionPriceMode
    HAS_ENHANCED_VBT = True
except ImportError:
    try:
        from ashare_vectorbt_engine.engine.ashare_sim import EnhancedVectorBTEngine
        from ashare_vectorbt_engine.contracts import ExecutionAssumptions, ExecutionPriceMode
        HAS_ENHANCED_VBT = True
    except ImportError:
        HAS_ENHANCED_VBT = False


def _standalone_vectorized_simulate(
    bars_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    initial_cash: float = 10000000.0,
    fee_rate: float = 0.0003,
    stamp_duty: float = 0.0005,
    slippage: float = 0.0002,
) -> Dict[str, Any]:
    """
    独立轻量向量化组合测算引擎 (当未接入 EnhancedVectorBTEngine 时自举运行)
    严格遵循 A 股单边印花税与双边佣金滑点磨损。
    """
    time_col = "trade_time" if "trade_time" in bars_df.columns else "trade_date"
    pivoted_close = bars_df.pivot(index=time_col, columns="symbol", values="close").ffill().bfill()
    asset_rets = pivoted_close.pct_change().fillna(0.0)

    # 对齐索引
    weights = weights_df.reindex(pivoted_close.index).fillna(0.0)

    # 换手率计算: sum(|w_t - w_{t-1}|)
    weight_diff = weights.diff().abs().fillna(weights.iloc[0].abs())
    daily_turnover = weight_diff.sum(axis=1)

    # 交易磨损: 买入(佣金+滑点), 卖出(佣金+滑点+印花税)
    # 假设平均买卖各半
    avg_friction_rate = fee_rate + slippage + (stamp_duty / 2.0)
    friction_drag = daily_turnover * avg_friction_rate

    # 日度毛收益与净收益
    gross_port_rets = (weights.shift(1).fillna(0.0) * asset_rets).sum(axis=1)
    net_port_rets = gross_port_rets - friction_drag

    equity_curve = (1.0 + net_port_rets).cumprod()
    total_return = float(equity_curve.iloc[-1] - 1.0) if len(equity_curve) > 0 else 0.0

    n_bars = len(net_port_rets)
    ann_factor = 252.0 if n_bars > 1 else 1.0
    ann_return = float(equity_curve.iloc[-1] ** (ann_factor / max(n_bars, 1)) - 1.0) if len(equity_curve) > 0 else 0.0

    ret_std = float(net_port_rets.std())
    sharpe = float(net_port_rets.mean() / ret_std * np.sqrt(ann_factor)) if ret_std > 1e-8 else 0.0

    cummax = equity_curve.cummax()
    drawdowns = (equity_curve - cummax) / cummax
    max_dd = float(abs(drawdowns.min())) if len(drawdowns) > 0 else 0.0
    calmar = float(ann_return / max_dd) if max_dd > 1e-4 else 0.0

    total_turnover = float(daily_turnover.sum())
    ann_turnover = float(daily_turnover.mean() * ann_factor)
    total_fees = float(initial_cash * (friction_drag.sum()))

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "annualized_turnover": ann_turnover,
        "total_fees_paid": total_fees,
        "total_impact_paid": float(total_fees * 0.3),
        "filled_orders": int((weight_diff > 0.001).sum().sum()),
    }


def execute_single_backtest_task(
    market_data: Union[pa.Table, pd.DataFrame],
    strategy_type: str,
    params: Dict[str, Any],
    initial_cash: float = 10000000.0,
) -> Dict[str, Any]:
    """
    单个参数组合的独立撮合回测逻辑 (支持进程内复用或 Ray 远程调用)
    """
    if isinstance(market_data, pa.Table):
        bars_df = market_data.to_pandas()
    else:
        bars_df = market_data

    time_col = "trade_time" if "trade_time" in bars_df.columns else "trade_date"
    pivoted_close = bars_df.pivot(index=time_col, columns="symbol", values="close").ffill().bfill()
    symbols = sorted(list(pivoted_close.columns))
    timeline = sorted(list(pivoted_close.index))

    if strategy_type == "trend":
        # 双均线趋势策略
        fast = int(params.get("fast_period", 10))
        slow = int(params.get("slow_period", 60))
        ma_f = pivoted_close.rolling(fast, min_periods=3).mean()
        ma_s = pivoted_close.rolling(slow, min_periods=5).mean()
        signal = (ma_f > ma_s).astype(float)
        # 归一化权重
        weight_sum = signal.sum(axis=1).replace(0.0, np.nan)
        weights_df = signal.div(weight_sum, axis=0).fillna(0.0) * 0.95

    elif strategy_type == "mean_reversion":
        # 布林带均值回归
        period = int(params.get("period", 20))
        std_dev = float(params.get("std_dev", 2.0))
        mid = pivoted_close.rolling(period, min_periods=5).mean()
        std = pivoted_close.rolling(period, min_periods=5).std().fillna(0.01)
        lower = mid - std_dev * std
        signal = (pivoted_close < lower).astype(float)
        weight_sum = signal.sum(axis=1).replace(0.0, np.nan)
        weights_df = signal.div(weight_sum, axis=0).fillna(0.0) * 0.90

    else:
        # 等权基准
        weights_df = pd.DataFrame(1.0 / len(symbols), index=timeline, columns=symbols)

    if HAS_ENHANCED_VBT:
        engine = EnhancedVectorBTEngine(ExecutionAssumptions(
            initial_cash=initial_cash,
            price_mode=ExecutionPriceMode.VWAP,
            enable_impact_cost=True,
        ))
        res = engine.run(bars_df=bars_df, target_weights_df=weights_df)
        return {
            "params": params,
            "total_return": float(res.total_return),
            "annualized_return": float(res.metrics["annualized_return"]),
            "sharpe_ratio": float(res.sharpe_ratio),
            "max_drawdown": float(res.max_drawdown),
            "calmar_ratio": float(res.metrics["calmar_ratio"]),
            "annualized_turnover": float(res.metrics["annualized_turnover"]),
            "total_fees_paid": float(res.metrics["total_fees_paid"]),
            "total_impact_paid": float(res.metrics.get("total_market_impact_paid", 0.0)),
            "filled_orders": int(res.metrics["filled_orders_count"]),
        }
    else:
        perf = _standalone_vectorized_simulate(bars_df, weights_df, initial_cash)
        perf["params"] = params
        return perf


if HAS_RAY:
    @ray.remote
    def ray_backtest_task(
        market_data_ref: Any,
        strategy_type: str,
        params: Dict[str, Any],
        initial_cash: float = 10000000.0,
    ) -> Dict[str, Any]:
        """Ray 远程无状态执行任务，利用 Plasma 零拷贝读取 market_data_ref"""
        table = ray.get(market_data_ref) if isinstance(market_data_ref, ray.ObjectRef) else market_data_ref
        return execute_single_backtest_task(table, strategy_type, params, initial_cash)
