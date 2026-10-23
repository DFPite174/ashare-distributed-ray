"""
分布式 AFML 蒙特卡洛与参数高原并行模拟 Worker (Distributed Audit Worker)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union
import numpy as np

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False


def execute_monte_carlo_batch_task(
    returns_arr: np.ndarray,
    batch_size: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    单个批次的蒙特卡洛路径重抽样与极值分布统计
    """
    rng = np.random.default_rng(seed)
    n_days = len(returns_arr)
    sim_sharpes = []
    sim_max_dds = []

    for _ in range(batch_size):
        # 几何布朗运动或有放回自举抽样
        sampled_rets = rng.choice(returns_arr, size=n_days, replace=True)
        eq_curve = np.cumprod(1.0 + sampled_rets)
        total_ret = eq_curve[-1] - 1.0

        ret_std = float(np.std(sampled_rets))
        sh = float(np.mean(sampled_rets) / ret_std * np.sqrt(252.0)) if ret_std > 1e-8 else 0.0

        cummax = np.maximum.accumulate(eq_curve)
        dd = (eq_curve - cummax) / cummax
        mdd = float(abs(np.min(dd)))

        sim_sharpes.append(sh)
        sim_max_dds.append(mdd)

    return {
        "batch_size": batch_size,
        "seed": seed,
        "mean_sim_sharpe": float(np.mean(sim_sharpes)),
        "p05_sharpe": float(np.percentile(sim_sharpes, 5)),
        "p95_sharpe": float(np.percentile(sim_sharpes, 95)),
        "mean_max_dd": float(np.mean(sim_max_dds)),
        "p95_max_dd": float(np.percentile(sim_max_dds, 95)),
    }


if HAS_RAY:
    @ray.remote
    def ray_monte_carlo_task(
        returns_ref: Any,
        batch_size: int = 100,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Ray 远程无状态执行任务，利用 Plasma 零拷贝读取 returns_ref"""
        arr = ray.get(returns_ref) if isinstance(returns_ref, ray.ObjectRef) else returns_ref
        return execute_monte_carlo_batch_task(arr, batch_size=batch_size, seed=seed)
