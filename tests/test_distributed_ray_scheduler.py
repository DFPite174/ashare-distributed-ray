"""
分布式调度与批量并发计算单元测试
"""

import pytest
import numpy as np
import pandas as pd
from ashare_distributed_ray.scheduler.ray_scheduler import DistributedRayScheduler


def test_distributed_grid_backtest():
    # 构造测试行情数据
    dates = pd.date_range("2024-01-01", periods=60, freq="B").strftime("%Y-%m-%d").tolist()
    bars = []
    np.random.seed(42)
    p = 20.0
    for d in dates:
        p *= (1.0 + np.random.normal(0.001, 0.02))
        bars.append({
            "trade_date": d,
            "symbol": "000001.SZ",
            "open": p * 0.99,
            "high": p * 1.02,
            "low": p * 0.98,
            "close": p,
            "volume": 100000,
            "amount": 2000000.0,
        })
    df = pd.DataFrame(bars)

    scheduler = DistributedRayScheduler(num_cpus=2)
    scheduler.table_manager.register_dataframe("test_bars", df, broadcast_to_ray=False)

    param_grid = [
        {"fast_period": 5, "slow_period": 20},
        {"fast_period": 10, "slow_period": 30},
        {"fast_period": 15, "slow_period": 45},
    ]

    results = scheduler.run_distributed_grid_backtest(
        table_name="test_bars",
        strategy_type="trend",
        param_grid=param_grid,
    )

    assert len(results) == 3
    for r in results:
        assert "sharpe_ratio" in r
        assert "total_return" in r
        assert "annualized_return" in r


def test_distributed_factor_and_monte_carlo():
    dates = pd.date_range("2024-01-01", periods=40, freq="B").strftime("%Y-%m-%d").tolist()
    bars = []
    for d in dates:
        bars.append({
            "trade_date": d,
            "symbol": "000001.SZ",
            "close": 10.0 + np.random.normal(0, 0.5),
            "amount": 1000000.0,
        })
    df = pd.DataFrame(bars)

    scheduler = DistributedRayScheduler(num_cpus=2)
    scheduler.table_manager.register_dataframe("factor_bars", df, broadcast_to_ray=False)

    factor_tasks = [
        ("momentum", {"window": 10}),
        ("volatility", {"window": 10}),
        ("rsi", {"window": 14}),
    ]

    f_res = scheduler.run_distributed_factor_batch(
        table_name="factor_bars",
        factor_tasks=factor_tasks,
    )
    assert len(f_res) == 3
    assert set([r["factor_name"] for r in f_res]) == {"momentum", "volatility", "rsi"}

    # 蒙特卡洛分布式测试
    rets = np.random.normal(0.001, 0.015, 100)
    mc_res = scheduler.run_distributed_monte_carlo(rets, total_simulations=200, batch_size=50)

    assert mc_res["total_simulations"] == 200
    assert "global_mean_sharpe" in mc_res
    assert "global_mean_mdd" in mc_res
