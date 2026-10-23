# A-Share Distributed Ray Engine (`ashare-distributed-ray`)

> 机构级 A 股分布式多租户量化计算与零拷贝共享内存中枢 (Ray + PyArrow Plasma + DuckDB Vectorized Analytics)

---

## 🌟 核心工程特色

1. **零拷贝列式共享内存 (Zero-Copy Columnar IPC)**:
   - 基于 **Apache Arrow** 列式内存规范与 **Ray Plasma In-Memory Object Store**；
   - 彻底告别传统 Python 多进程/集群节点间重复反序列化 Pandas DataFrame 的巨大内存膨胀（Pickle Storm）与 GC 停顿，全市场历史数据在单节点内仅驻留一份只读内存，多 Worker 并发读取开销降至亚毫秒级。
2. **DuckDB 嵌入式向量化时序切片引擎**:
   - 结合 DuckDB 向量化执行器，直接对内存中的 PyArrow Table 进行零拷贝 SQL 切片与复杂聚合过滤（如 `slice_market_bars`），单次切片耗时低于 1 毫秒。
3. **基于 DRF 的多租户资源配额与准入控制 (Multi-Tenant Quota Guard)**:
   - 借鉴伯克利 *Dominant Resource Fairness (NSDI 2011)* 理论，对租户级 CPU 核心数、内存消耗量和并发作业数执行硬限额监控；
   - 动态准入控制（Admission Control），防止单研究员提交的海量超维参数网格拖垮集群。
4. **作业生命周期跟踪与硬超时熔断看门狗 (Job Lifecycle Watchdog)**:
   - 全链路追踪作业状态：`SUBMITTED` $\to$ `QUEUED` $\to$ `RUNNING` $\to$ `COMPLETED` / `FAILED` / `TIMEOUT_KILLED`；
   - 周期性巡检并对超出最大容忍时限的死锁/死循环 Worker 执行硬熔断释放算力。
5. **高并发投研作业一键并行化**:
   - **分布式参数网格回测 (`run_distributed_grid_backtest`)**：向量化撮合内核超维并行搜索；
   - **分布式批量因子计算 (`run_distributed_factor_batch`)**：全市场截面因子并行打分；
   - **分布式 AFML 蒙特卡洛审计 (`run_distributed_monte_carlo`)**：千条路径自举抽样秒级归约。

---

## 🛠️ 快速上手

```python
import pandas as pd
from ashare_distributed_ray import DistributedRayScheduler

# 1. 初始化调度器与共享内存
scheduler = DistributedRayScheduler(num_cpus=4, auto_init_ray=True)

# 2. 注册市场行情数据至零拷贝共享内存
df_bars = pd.read_parquet("market_bars.parquet")
scheduler.table_manager.register_dataframe("market_bars", df_bars)

# 3. 分布式并发执行参数网格回测
param_grid = [
    {"fast_period": 5, "slow_period": 20},
    {"fast_period": 10, "slow_period": 30},
    {"fast_period": 15, "slow_period": 60},
]

results = scheduler.run_distributed_grid_backtest(
    table_name="market_bars",
    strategy_type="trend",
    param_grid=param_grid,
    tenant_id="researcher_alpha",
)

for r in results:
    print(r["params"], "Sharpe:", r["sharpe_ratio"])
```

---

## 🧪 单元测试

```bash
pytest tests/
```
