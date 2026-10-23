"""
A-Share Distributed Ray Engine - 任务与多租户契约层 (Distributed Compute Contracts)
参考:
- Ray: A Distributed Framework for Emerging AI Applications (Moritz et al., OSDI 2018)
- Dominant Resource Fairness: Fair Allocation of Multiple Resource Types (Ghodsi et al., NSDI 2011)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import uuid


class JobPriority(str, Enum):
    """任务优先级 (用于多租户优先级队列调度)"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def weight(self) -> int:
        weights = {"LOW": 1, "MEDIUM": 5, "HIGH": 10, "CRITICAL": 50}
        return weights[self.value]


class JobStatus(str, Enum):
    """分布式任务状态机"""
    SUBMITTED = "SUBMITTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT_KILLED = "TIMEOUT_KILLED"
    CANCELLED = "CANCELLED"


class JobType(str, Enum):
    """分布式任务类型"""
    FACTOR_COMPUTE = "FACTOR_COMPUTE"                # 批量因子截面计算
    FACTOR_CROSS_SECTION = "FACTOR_CROSS_SECTION"    # 截面计算别名
    PARAMETER_GRID_BACKTEST = "PARAMETER_GRID_BACKTEST"  # 参数网格并行回测
    STRATEGY_GRID_SEARCH = "STRATEGY_GRID_SEARCH"        # 网格搜索别名
    MONTE_CARLO_AUDIT = "MONTE_CARLO_AUDIT"          # AFML 蒙特卡洛/置换检验并行审计
    CUSTOM = "CUSTOM"                                # 用户自定义计算任务


@dataclass(frozen=True)
class TenantQuota:
    """多租户资源配额配置 (Dominant Resource Fairness 约束)"""
    tenant_id: str
    tenant_name: str
    max_cpu_cores: float = 8.0              # 该租户可占用的最大 CPU 核心数
    max_memory_mb: float = 16384.0          # 该租户可占用的最大内存 (MB)
    max_concurrent_jobs: int = 20           # 最大并发任务数
    max_job_duration_sec: float = 600.0     # 单任务硬超时时限 (秒)


@dataclass
class ComputeJobSpec:
    """分布式计算作业定义规范"""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = "default_researcher"
    job_type: JobType = JobType.CUSTOM
    priority: JobPriority = JobPriority.MEDIUM
    cpu_limit: float = 1.0                  # 所需 CPU 核心数
    memory_limit_mb: float = 1024.0         # 预估内存消耗
    timeout_sec: float = 300.0              # 任务超时时间
    func_name: str = ""                     # 目标执行函数名或标识
    payload: Dict[str, Any] = field(default_factory=dict) # 序列化参数载荷
    submitted_at: datetime = field(default_factory=datetime.now)


@dataclass
class ComputeJobResult:
    """分布式作业执行产出与指标"""
    job_id: str
    tenant_id: str
    status: JobStatus
    return_value: Any = None
    error_msg: Optional[str] = None
    execution_time_sec: float = 0.0
    cpu_time_sec: float = 0.0
    memory_peak_mb: float = 0.0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    worker_node_id: Optional[str] = None

    @property
    def error_message(self) -> Optional[str]:
        return self.error_msg
