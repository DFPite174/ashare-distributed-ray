"""
DuckDB 嵌入式零拷贝列式计算中枢 (DuckDB In-Memory Analytical Engine)
参考:
- Raasveldt & Mühleisen, "DuckDB: an Embeddable Analytical Database" (VLDB 2019)
- Zero-Copy Apache Arrow Integration with DuckDB
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Any
import duckdb
import pyarrow as pa
import pandas as pd

from ashare_distributed_ray.memory.shared_table import SharedArrowTableManager


class DuckDBSharedCatalog:
    """
    基于 DuckDB 向量化执行引擎与 PyArrow 零拷贝内存视图，
    提供亚毫秒级的全市场股票切片过滤与复杂时序聚合能力。
    """

    def __init__(self, table_manager: Optional[SharedArrowTableManager] = None):
        self.table_manager = table_manager or SharedArrowTableManager()
        self.con = duckdb.connect(database=":memory:")
        self._registered_tables: set[str] = set()

    def register_arrow_table(self, table_name: str, arrow_table: Optional[pa.Table] = None):
        tbl = arrow_table or self.table_manager.get_arrow_table(table_name)
        self.con.register(table_name, tbl)
        self._registered_tables.add(table_name)

    def query(self, sql: str) -> pa.Table:
        """执行 SQL 并零拷贝返回 PyArrow Table 结果"""
        return self.con.sql(sql).to_arrow_table()

    def query_df(self, sql: str) -> pd.DataFrame:
        """执行 SQL 并返回 Pandas DataFrame"""
        return self.con.sql(sql).df()

    def slice_market_bars(
        self,
        table_name: str = "bars",
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if table_name not in self._registered_tables:
            self.register_arrow_table(table_name)

        col_str = ", ".join(columns) if columns else "*"
        conditions = []

        tbl = self.table_manager.get_arrow_table(table_name)
        avail_cols = set(tbl.column_names)

        if symbols and "symbol" in avail_cols:
            quoted_syms = ", ".join([f"'{s}'" for s in symbols])
            conditions.append(f"symbol IN ({quoted_syms})")

        date_col = "trade_time" if "trade_time" in avail_cols else ("trade_date" if "trade_date" in avail_cols else None)
        if date_col:
            if start_date:
                val = start_date if date_col == "trade_time" else start_date[:10]
                conditions.append(f"{date_col} >= '{val}'")
            if end_date:
                val = end_date if date_col == "trade_time" else end_date[:10]
                conditions.append(f"{date_col} <= '{val}'")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT {col_str} FROM {table_name} {where_clause}"

        return self.query_df(sql)

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass
