"""
零拷贝列式共享内存中枢 (Zero-Copy Columnar Shared Memory)
参考:
- Apache Arrow: A Cross-Language Development Platform for In-Memory Analytics
- Ray Plasma In-Memory Object Store Architecture
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False


class SharedArrowTableManager:
    """
    负责将市场数据转化为 PyArrow 列式内存结构并放置在共享内存中，
    实现多 Worker 进程间的零拷贝 (Zero-Copy) 广播与并发切片。
    """

    def __init__(self):
        self._local_tables: Dict[str, pa.Table] = {}
        self._ray_refs: Dict[str, Any] = {}

    def register_dataframe(
        self,
        name: str,
        df: pd.DataFrame,
        broadcast_to_ray: bool = True,
    ) -> pa.Table:
        """
        将 Pandas DataFrame 转换为不可变 PyArrow Table 并注册进共享内存
        """
        table = pa.Table.from_pandas(df, preserve_index=False)
        self._local_tables[name] = table

        if broadcast_to_ray and HAS_RAY and ray.is_initialized():
            # 放入 Ray 节点级 Plasma 共享内存对象存储 (Zero-Copy 广播)
            self._ray_refs[name] = ray.put(table)

        return table

    def register_parquet_file(
        self,
        name: str,
        file_path: Union[str, Path],
        columns: Optional[List[str]] = None,
        broadcast_to_ray: bool = True,
    ) -> pa.Table:
        """
        通过内存映射快速加载 Parquet 文件为 PyArrow Table
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Parquet file not found: {p}")

        table = pq.read_table(p, columns=columns, memory_map=True)
        self._local_tables[name] = table

        if broadcast_to_ray and HAS_RAY and ray.is_initialized():
            self._ray_refs[name] = ray.put(table)

        return table

    def get_arrow_table(self, name: str) -> pa.Table:
        """获取本地注册的 PyArrow Table"""
        if name not in self._local_tables:
            raise KeyError(f"Table '{name}' not found in shared table registry")
        return self._local_tables[name]

    def get_table(self, name: str) -> Optional[pa.Table]:
        """获取本地注册的 PyArrow Table (安全返回 None)"""
        return self._local_tables.get(name)

    def broadcast_to_ray(self, name: str) -> Any:
        """广播指定的表至 Ray Plasma 对象存储"""
        return self.get_ray_ref(name)

    def get_ray_ref(self, name: str) -> Any:
        """获取 Ray Object Store 共享内存引用"""
        if name not in self._ray_refs:
            if name in self._local_tables and HAS_RAY and ray.is_initialized():
                self._ray_refs[name] = ray.put(self._local_tables[name])
            else:
                raise KeyError(f"Ray ObjectRef for '{name}' is not registered")
        return self._ray_refs[name]

    @staticmethod
    def zero_copy_to_pandas(table_or_record_batch: Union[pa.Table, pa.RecordBatch]) -> pd.DataFrame:
        try:
            return table_or_record_batch.to_pandas(zero_copy_only=True)
        except Exception:
            return table_or_record_batch.to_pandas()

    def get_memory_footprint(self, name: str) -> int:
        table = self.get_arrow_table(name)
        return table.nbytes
