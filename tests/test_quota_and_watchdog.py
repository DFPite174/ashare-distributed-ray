"""
多租户资源配额与超时看门狗单元测试
"""

import pytest
import time
from ashare_distributed_ray.contracts import TenantQuota, ComputeJobSpec, JobStatus
from ashare_distributed_ray.quota.manager import TenantQuotaManager
from ashare_distributed_ray.quota.watchdog import JobLifecycleWatchdog


def test_tenant_quota_admission_and_allocation():
    mgr = TenantQuotaManager(cluster_total_cpu=8.0, cluster_total_memory_mb=16384.0)

    # 注册限制较严的小租户
    mgr.register_tenant(TenantQuota(
        tenant_id="intern_dev",
        tenant_name="Intern Researcher",
        max_cpu_cores=2.0,
        max_memory_mb=2048.0,
        max_concurrent_jobs=2,
    ))

    # Job 1: 正常接纳
    j1 = ComputeJobSpec(job_id="j1", tenant_id="intern_dev", cpu_limit=1.5, memory_limit_mb=1000.0)
    admitted1, _ = mgr.check_admission(j1)
    assert admitted1 is True
    assert mgr.allocate_resources(j1) is True

    # Job 2: CPU 超额 (1.5 + 1.0 > 2.0) -> 拦截
    j2 = ComputeJobSpec(job_id="j2", tenant_id="intern_dev", cpu_limit=1.0, memory_limit_mb=500.0)
    admitted2, reason = mgr.check_admission(j2)
    assert admitted2 is False
    assert "CPU quota exceeded" in reason

    # 释放 Job 1
    mgr.release_resources(j1)
    usage = mgr.get_usage("intern_dev")
    assert usage.active_cpu_cores == 0.0

    # Job 2 再次准入 -> 成功
    assert mgr.allocate_resources(j2) is True


def test_job_lifecycle_and_timeout_watchdog():
    mgr = TenantQuotaManager()
    watchdog = JobLifecycleWatchdog(mgr)

    # 构造极短超时的 Job (0.1秒)
    spec = ComputeJobSpec(job_id="test_timeout_job", tenant_id="default_researcher", timeout_sec=0.1)
    watchdog.record_job_start(spec)

    # 初始状态 RUNNING
    res = watchdog.get_job_result("test_timeout_job")
    assert res.status == JobStatus.RUNNING

    time.sleep(0.15)
    killed = watchdog.check_timeouts()

    assert "test_timeout_job" in killed
    killed_res = watchdog.get_job_result("test_timeout_job")
    assert killed_res.status == JobStatus.TIMEOUT_KILLED
    assert "Hard timeout" in (killed_res.error_message or "")
