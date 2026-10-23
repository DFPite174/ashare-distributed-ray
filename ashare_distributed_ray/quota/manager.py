"""
多租户资源配额与准入控制中枢 (Multi-Tenant Resource Quota Manager)
参考:
- Ghodsi et al., "Dominant Resource Fairness: Fair Allocation of Multiple Resource Types" (NSDI 2011)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import threading
from ashare_distributed_ray.contracts import TenantQuota, ComputeJobSpec, JobStatus


@dataclass
class TenantResourceUsage:
    """实时租户资源占用记录"""
    active_cpu_cores: float = 0.0
    active_memory_mb: float = 0.0
    active_jobs_count: int = 0
    active_job_ids: Set[str] = field(default_factory=set)


class TenantQuotaManager:
    """
    基于 DRF (Dominant Resource Fairness) 思想的租户资源守卫，
    防止多研究员并发提交大规模参数网格时单点爆破集群内存与 CPU。
    """

    def __init__(self, cluster_total_cpu: float = 16.0, cluster_total_memory_mb: float = 32768.0):
        self.cluster_total_cpu = cluster_total_cpu
        self.cluster_total_memory_mb = cluster_total_memory_mb
        self._quotas: Dict[str, TenantQuota] = {}
        self._usages: Dict[str, TenantResourceUsage] = {}
        self._lock = threading.RLock()

        self.register_tenant(TenantQuota(
            tenant_id="default_researcher",
            tenant_name="Default Quant Researcher",
            max_cpu_cores=min(8.0, cluster_total_cpu),
            max_memory_mb=min(16384.0, cluster_total_memory_mb),
            max_concurrent_jobs=30,
            max_job_duration_sec=600.0,
        ))

    def register_tenant(self, quota: TenantQuota):
        with self._lock:
            self._quotas[quota.tenant_id] = quota
            if quota.tenant_id not in self._usages:
                self._usages[quota.tenant_id] = TenantResourceUsage()

    def get_quota(self, tenant_id: str) -> TenantQuota:
        with self._lock:
            if tenant_id not in self._quotas:
                self.register_tenant(TenantQuota(
                    tenant_id=tenant_id,
                    tenant_name=f"Researcher_{tenant_id}",
                    max_cpu_cores=4.0,
                    max_memory_mb=8192.0,
                ))
            return self._quotas[tenant_id]

    def get_usage(self, tenant_id: str) -> TenantResourceUsage:
        with self._lock:
            if tenant_id not in self._usages:
                self._usages[tenant_id] = TenantResourceUsage()
            return self._usages[tenant_id]

    def check_admission(self, job_spec: ComputeJobSpec) -> Tuple[bool, str]:
        with self._lock:
            quota = self.get_quota(job_spec.tenant_id)
            usage = self.get_usage(job_spec.tenant_id)

            if usage.active_jobs_count >= quota.max_concurrent_jobs:
                return False, f"Tenant '{job_spec.tenant_id}' reached max concurrent jobs ({quota.max_concurrent_jobs})"

            if usage.active_cpu_cores + job_spec.cpu_limit > quota.max_cpu_cores:
                return False, f"Tenant '{job_spec.tenant_id}' CPU quota exceeded ({usage.active_cpu_cores + job_spec.cpu_limit:.1f} > {quota.max_cpu_cores:.1f})"

            if usage.active_memory_mb + job_spec.memory_limit_mb > quota.max_memory_mb:
                return False, f"Tenant '{job_spec.tenant_id}' Memory quota exceeded ({usage.active_memory_mb + job_spec.memory_limit_mb:.1f}MB > {quota.max_memory_mb:.1f}MB)"

            return True, "ADMITTED"

    def allocate_resources(self, job_spec: ComputeJobSpec) -> bool:
        with self._lock:
            admitted, reason = self.check_admission(job_spec)
            if not admitted:
                return False

            usage = self._usages[job_spec.tenant_id]
            usage.active_cpu_cores += job_spec.cpu_limit
            usage.active_memory_mb += job_spec.memory_limit_mb
            usage.active_jobs_count += 1
            usage.active_job_ids.add(job_spec.job_id)
            return True

    def release_resources(self, job_spec: ComputeJobSpec):
        with self._lock:
            usage = self.get_usage(job_spec.tenant_id)
            if job_spec.job_id in usage.active_job_ids:
                usage.active_job_ids.remove(job_spec.job_id)
                usage.active_cpu_cores = max(0.0, usage.active_cpu_cores - job_spec.cpu_limit)
                usage.active_memory_mb = max(0.0, usage.active_memory_mb - job_spec.memory_limit_mb)
                usage.active_jobs_count = max(0, usage.active_jobs_count - 1)
