"""
分布式 Ray 调度中枢与多租户分发器 (Distributed Ray Scheduler & Dispatcher)
参考:
- Moritz et al., "Ray: A Distributed Framework for Emerging AI Applications" (OSDI 2018)
- Ghodsi et al., "Dominant Resource Fairness" (NSDI 2011)
"""

from __future__ import annotations
import concurrent.futures
from datetime import datetime
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import pyarrow as pa

try:
    import ray
    from ashare_distributed_ray.workers.backtest_worker import ray_backtest_task
    from ashare_distributed_ray.workers.factor_worker import ray_factor_task
    from ashare_distributed_ray.workers.audit_worker import ray_monte_carlo_task
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

from ashare_distributed_ray.contracts import (
    ComputeJobSpec,
    ComputeJobResult,
    JobPriority,
    JobStatus,
    JobType,
    TenantQuota,
)
from ashare_distributed_ray.memory.shared_table import SharedArrowTableManager
from ashare_distributed_ray.quota.manager import TenantQuotaManager
from ashare_distributed_ray.quota.watchdog import JobLifecycleWatchdog
from ashare_distributed_ray.workers.backtest_worker import execute_single_backtest_task
from ashare_distributed_ray.workers.factor_worker import execute_factor_batch_task
from ashare_distributed_ray.workers.audit_worker import execute_monte_carlo_batch_task


class DistributedRayScheduler:
    """
    企业级分布式投研调度总控:
    1. 管理 Ray 集群与本地进程池
    2. 基于 Plasma/Arrow 共享内存的零拷贝数据广播
    3. 多租户配额限制与超时硬熔断守卫
    4. 分布式参数网格回测、批量因子计算与 AFML 蒙特卡洛并行加速
    """

    def __init__(
        self,
        num_cpus: Optional[int] = None,
        auto_init_ray: bool = False,
        cluster_total_cpu: float = 16.0,
        cluster_total_memory_mb: float = 32768.0,
    ):
        self.num_cpus = num_cpus or max(1, os.cpu_count() or 4)
        self.table_manager = SharedArrowTableManager()
        self.quota_manager = TenantQuotaManager(
            cluster_total_cpu=cluster_total_cpu,
            cluster_total_memory_mb=cluster_total_memory_mb,
        )
        self.watchdog = JobLifecycleWatchdog(self.quota_manager)
        self._is_ray_active = False

        if auto_init_ray and HAS_RAY:
            self.init_cluster()

    def init_cluster(self, address: Optional[str] = None):
        """初始化连接外部 Ray 集群或就地启动本地 Ray 实例"""
        if not HAS_RAY:
            raise RuntimeError("Ray 库未安装，无法启动分布式集群模式")
        if not ray.is_initialized():
            if address:
                ray.init(address=address, ignore_reinit_error=True)
            else:
                ray.init(
                    num_cpus=self.num_cpus,
                    include_dashboard=False,
                    ignore_reinit_error=True,
                    logging_level="error",
                )
        self._is_ray_active = True

    def shutdown_cluster(self):
        """安全停用 Ray 集群"""
        if HAS_RAY and ray.is_initialized():
            ray.shutdown()
        self._is_ray_active = False

    def shutdown(self):
        """安全停用 Ray 集群 (别名)"""
        self.shutdown_cluster()

    @property
    def is_ray_active(self) -> bool:
        return bool(HAS_RAY and ray.is_initialized())

    def run_distributed_grid_backtest(
        self,
        table_name: str,
        strategy_type: str,
        param_grid: List[Dict[str, Any]],
        tenant_id: str = "default_researcher",
        initial_cash: float = 10000000.0,
        timeout_sec: float = 300.0,
    ) -> List[Dict[str, Any]]:
        """
        分布式参数网格回测执行入口
        将 PyArrow Table 注册进 Plasma 内存并并发分发网格任务
        """
        table = self.table_manager.get_table(table_name)
        if table is None:
            raise KeyError(f"行情表未在共享管理器中注册: {table_name}")

        job_spec = ComputeJobSpec(
            job_id=f"grid_bt_{int(time.time()*1000)}",
            tenant_id=tenant_id,
            job_type=JobType.STRATEGY_GRID_SEARCH,
            cpu_limit=float(len(param_grid)),
            timeout_sec=timeout_sec,
        )

        admitted, reason = self.quota_manager.check_admission(job_spec)
        if not admitted:
            raise PermissionError(f"网格回测任务准入拒绝: {reason}")

        self.quota_manager.allocate_resources(job_spec)
        self.watchdog.record_job_start(job_spec)

        start_time = time.time()
        results: List[Dict[str, Any]] = []

        try:
            if self.is_ray_active:
                # 1. 确保数据在 Ray Plasma 共享对象池中 (零拷贝广播)
                ray_ref = self.table_manager.get_ray_ref(table_name)
                if ray_ref is None:
                    ray_ref = self.table_manager.broadcast_to_ray(table_name)

                futures = [
                    ray_backtest_task.remote(ray_ref, strategy_type, params, initial_cash)
                    for params in param_grid
                ]
                results = ray.get(futures)

            else:
                # 2. 本地多线程/进程并发兜底
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_cpus) as executor:
                    fut_map = {
                        executor.submit(execute_single_backtest_task, table, strategy_type, p, initial_cash): p
                        for p in param_grid
                    }
                    for fut in concurrent.futures.as_completed(fut_map):
                        results.append(fut.result())

            self.watchdog.record_job_completion(job_spec.job_id, return_value=results, status=JobStatus.COMPLETED)
            return results

        except Exception as e:
            self.watchdog.record_job_completion(job_spec.job_id, error_msg=str(e), status=JobStatus.FAILED)
            raise e

    def run_distributed_factor_batch(
        self,
        table_name: str,
        factor_tasks: List[Tuple[str, Optional[Dict[str, Any]]]],
        tenant_id: str = "default_researcher",
        timeout_sec: float = 300.0,
    ) -> List[Dict[str, Any]]:
        """
        分布式批量因子截面计算
        """
        table = self.table_manager.get_table(table_name)
        if table is None:
            raise KeyError(f"行情表未在共享管理器中注册: {table_name}")

        job_spec = ComputeJobSpec(
            job_id=f"factor_batch_{int(time.time()*1000)}",
            tenant_id=tenant_id,
            job_type=JobType.FACTOR_CROSS_SECTION,
            cpu_limit=float(len(factor_tasks)),
            timeout_sec=timeout_sec,
        )

        admitted, reason = self.quota_manager.check_admission(job_spec)
        if not admitted:
            raise PermissionError(f"因子批量计算准入拒绝: {reason}")

        self.quota_manager.allocate_resources(job_spec)
        self.watchdog.record_job_start(job_spec)

        try:
            results = []
            if self.is_ray_active:
                ray_ref = self.table_manager.get_ray_ref(table_name)
                if ray_ref is None:
                    ray_ref = self.table_manager.broadcast_to_ray(table_name)

                futures = [
                    ray_factor_task.remote(ray_ref, factor_name, params)
                    for factor_name, params in factor_tasks
                ]
                results = ray.get(futures)

            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_cpus) as executor:
                    fut_map = {
                        executor.submit(execute_factor_batch_task, table, fname, p): fname
                        for fname, p in factor_tasks
                    }
                    for fut in concurrent.futures.as_completed(fut_map):
                        results.append(fut.result())

            self.watchdog.record_job_completion(job_spec.job_id, return_value=results, status=JobStatus.COMPLETED)
            return results

        except Exception as e:
            self.watchdog.record_job_completion(job_spec.job_id, error_msg=str(e), status=JobStatus.FAILED)
            raise e

    def run_distributed_monte_carlo(
        self,
        returns_arr: Optional[Union[np.ndarray, List[float]]] = None,
        total_simulations: int = 1000,
        batch_size: int = 100,
        tenant_id: str = "default_researcher",
        timeout_sec: float = 300.0,
        returns: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        分布式 AFML 蒙特卡洛/置换检验并行模拟
        """
        ret_data = returns_arr if returns_arr is not None else returns
        if ret_data is None:
            raise ValueError("Must provide returns_arr or returns")
        arr = np.asarray(ret_data, dtype=np.float64)
        n_batches = int(np.ceil(total_simulations / batch_size))

        job_spec = ComputeJobSpec(
            job_id=f"monte_carlo_{int(time.time()*1000)}",
            tenant_id=tenant_id,
            job_type=JobType.MONTE_CARLO_AUDIT,
            cpu_limit=float(n_batches),
            timeout_sec=timeout_sec,
        )

        admitted, reason = self.quota_manager.check_admission(job_spec)
        if not admitted:
            raise PermissionError(f"蒙特卡洛模拟准入拒绝: {reason}")

        self.quota_manager.allocate_resources(job_spec)
        self.watchdog.record_job_start(job_spec)

        try:
            batch_results = []
            if self.is_ray_active:
                returns_ref = ray.put(arr)
                futures = [
                    ray_monte_carlo_task.remote(returns_ref, batch_size=batch_size, seed=42 + i * 17)
                    for i in range(n_batches)
                ]
                batch_results = ray.get(futures)

            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_cpus) as executor:
                    fut_map = {
                        executor.submit(execute_monte_carlo_batch_task, arr, batch_size, 42 + i * 17): i
                        for i in range(n_batches)
                    }
                    for fut in concurrent.futures.as_completed(fut_map):
                        batch_results.append(fut.result())

            all_sharpes = [b["mean_sim_sharpe"] for b in batch_results]
            all_mdds = [b["mean_max_dd"] for b in batch_results]

            summary = {
                "total_simulations": n_batches * batch_size,
                "n_batches": n_batches,
                "global_mean_sharpe": float(np.mean(all_sharpes)),
                "p05_sharpe": float(np.percentile(all_sharpes, 5)),
                "p95_sharpe": float(np.percentile(all_sharpes, 95)),
                "global_mean_mdd": float(np.mean(all_mdds)),
                "p95_mdd": float(np.percentile(all_mdds, 95)),
                "batch_results": batch_results,
            }

            self.watchdog.record_job_completion(job_spec.job_id, return_value=summary, status=JobStatus.COMPLETED)
            return summary

        except Exception as e:
            self.watchdog.record_job_completion(job_spec.job_id, error_msg=str(e), status=JobStatus.FAILED)
            raise e
