"""
任务生命周期跟踪与硬超时看门狗 (Job Lifecycle Watchdog)
"""

from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Optional, Any
import threading
import time

from ashare_distributed_ray.contracts import ComputeJobSpec, ComputeJobResult, JobStatus
from ashare_distributed_ray.quota.manager import TenantQuotaManager


class JobLifecycleWatchdog:
    """
    负责监控计算任务生命周期、探测死锁或死循环任务，
    并在任务超出最大容忍时限时执行硬熔断 (Hard Timeout Kill)，及时回收集群算力。
    """

    def __init__(self, quota_manager: TenantQuotaManager):
        self.quota_manager = quota_manager
        self._active_jobs: Dict[str, ComputeJobSpec] = {}
        self._job_start_times: Dict[str, float] = {}
        self._job_results: Dict[str, ComputeJobResult] = {}
        self._lock = threading.RLock()

    def record_job_start(self, spec: ComputeJobSpec):
        with self._lock:
            self._active_jobs[spec.job_id] = spec
            self._job_start_times[spec.job_id] = time.time()
            self._job_results[spec.job_id] = ComputeJobResult(
                job_id=spec.job_id,
                tenant_id=spec.tenant_id,
                status=JobStatus.RUNNING,
                started_at=datetime.now(),
            )

    def record_job_completion(
        self,
        job_id: str,
        return_value: Any = None,
        error_msg: Optional[str] = None,
        status: JobStatus = JobStatus.COMPLETED,
    ) -> ComputeJobResult:
        with self._lock:
            spec = self._active_jobs.pop(job_id, None)
            start_t = self._job_start_times.pop(job_id, time.time())
            elapsed = time.time() - start_t

            if spec is not None:
                self.quota_manager.release_resources(spec)

            res = self._job_results.get(job_id)
            if res is None:
                res = ComputeJobResult(
                    job_id=job_id,
                    tenant_id=spec.tenant_id if spec else "unknown",
                    status=status,
                )

            res.status = status
            res.return_value = return_value
            res.error_msg = error_msg
            res.execution_time_sec = elapsed
            res.finished_at = datetime.now()
            self._job_results[job_id] = res
            return res

    def check_timeouts(self) -> List[str]:
        killed_jobs = []
        now = time.time()

        with self._lock:
            for job_id, spec in list(self._active_jobs.items()):
                start_t = self._job_start_times.get(job_id, now)
                duration = now - start_t
                if duration > spec.timeout_sec:
                    self.record_job_completion(
                        job_id=job_id,
                        error_msg=f"Hard timeout: Job exceeded timeout limit of {spec.timeout_sec:.1f}s (ran for {duration:.1f}s)",
                        status=JobStatus.TIMEOUT_KILLED,
                    )
                    killed_jobs.append(job_id)

        return killed_jobs

    def get_job_result(self, job_id: str) -> Optional[ComputeJobResult]:
        with self._lock:
            return self._job_results.get(job_id)
