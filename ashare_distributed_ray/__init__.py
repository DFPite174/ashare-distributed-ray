"""
ashare-distributed-ray: 企业级 Ray 分布式多租户计算引擎 (Enterprise Ray Distributed Multi-Tenant Compute)
"""

from ashare_distributed_ray.contracts import (
    TenantQuota,
    ComputeJobSpec,
    ComputeJobResult,
    JobPriority,
    JobStatus,
    JobType,
)
from ashare_distributed_ray.memory.shared_table import SharedArrowTableManager
from ashare_distributed_ray.memory.duckdb_catalog import DuckDBSharedCatalog
from ashare_distributed_ray.quota.manager import TenantQuotaManager, TenantResourceUsage
from ashare_distributed_ray.quota.watchdog import JobLifecycleWatchdog
from ashare_distributed_ray.scheduler.ray_scheduler import DistributedRayScheduler

__version__ = "1.0.0"

__all__ = [
    "TenantQuota",
    "ComputeJobSpec",
    "ComputeJobResult",
    "JobPriority",
    "JobStatus",
    "JobType",
    "SharedArrowTableManager",
    "DuckDBSharedCatalog",
    "TenantQuotaManager",
    "TenantResourceUsage",
    "JobLifecycleWatchdog",
    "DistributedRayScheduler",
]
