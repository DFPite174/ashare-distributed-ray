"""
分布式因子截面特征计算 Worker (Distributed Factor Worker)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
import pyarrow as pa

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False


def execute_factor_batch_task(
    market_data: Union[pa.Table, pd.DataFrame],
    factor_name: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    单个因子的截面向量化计算逻辑
    """
    if isinstance(market_data, pa.Table):
        bars_df = market_data.to_pandas()
    else:
        bars_df = market_data

    time_col = "trade_time" if "trade_time" in bars_df.columns else "trade_date"
    pivoted_close = bars_df.pivot(index=time_col, columns="symbol", values="close").ffill().bfill()
    pivoted_amount = bars_df.pivot(index=time_col, columns="symbol", values="amount").ffill().bfill() if "amount" in bars_df.columns else pivoted_close
    p = params or {}

    if factor_name == "momentum":
        window = int(p.get("window", 20))
        res_df = pivoted_close.pct_change(window).fillna(0.0)

    elif factor_name == "volatility":
        window = int(p.get("window", 20))
        rets = pivoted_close.pct_change().fillna(0.0)
        res_df = rets.rolling(window, min_periods=5).std().fillna(0.02)

    elif factor_name == "rsi":
        window = int(p.get("window", 14))
        delta = pivoted_close.diff()
        gain = delta.clip(lower=0.0).rolling(window).mean()
        loss = (-delta.clip(upper=0.0)).rolling(window).mean()
        rs = gain / (loss + 1e-9)
        res_df = 100.0 - (100.0 / (1.0 + rs)).fillna(50.0)

    elif factor_name == "turnover_ratio":
        window = int(p.get("window", 20))
        res_df = pivoted_amount.rolling(window, min_periods=5).mean().fillna(1e7)

    else:
        # 默认收益率
        res_df = pivoted_close.pct_change().fillna(0.0)

    # 截面统计摘要
    latest_scores = res_df.iloc[-1].to_dict()
    mean_val = float(res_df.values.mean())
    std_val = float(res_df.values.std())

    return {
        "factor_name": factor_name,
        "params": p,
        "mean_value": mean_val,
        "std_value": std_val,
        "latest_top_symbol": max(latest_scores, key=latest_scores.get) if latest_scores else None,
        "latest_scores": latest_scores,
    }


if HAS_RAY:
    @ray.remote
    def ray_factor_task(
        market_data_ref: Any,
        factor_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ray 远程无状态执行任务，利用 Plasma 零拷贝读取 market_data_ref"""
        table = ray.get(market_data_ref) if isinstance(market_data_ref, ray.ObjectRef) else market_data_ref
        return execute_factor_batch_task(table, factor_name, params)
