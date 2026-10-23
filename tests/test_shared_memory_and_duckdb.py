"""
零拷贝列式共享内存与 DuckDB 嵌入式切片单元测试
"""

import pytest
import numpy as np
import pandas as pd
import pyarrow as pa

from ashare_distributed_ray.memory.shared_table import SharedArrowTableManager
from ashare_distributed_ray.memory.duckdb_catalog import DuckDBSharedCatalog


def test_shared_arrow_table_lifecycle():
    df = pd.DataFrame({
        "trade_date": ["2025-01-02", "2025-01-03", "2025-01-06"],
        "symbol": ["000001.SZ", "000001.SZ", "000001.SZ"],
        "close": [10.5, 10.8, 11.2],
        "volume": [100000.0, 120000.0, 150000.0],
    })

    mgr = SharedArrowTableManager()
    table = mgr.register_dataframe("test_bars", df, broadcast_to_ray=False)

    assert isinstance(table, pa.Table)
    assert table.num_rows == 3
    assert "close" in table.column_names

    # 验证零拷贝至 Pandas
    df_recovered = mgr.zero_copy_to_pandas(table)
    assert len(df_recovered) == 3
    assert abs(df_recovered["close"].iloc[0] - 10.5) < 1e-6
    assert mgr.get_memory_footprint("test_bars") > 0


def test_duckdb_zero_copy_query_and_slice():
    df = pd.DataFrame({
        "trade_date": ["2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"],
        "symbol": ["000001.SZ", "600519.SH", "000001.SZ", "600519.SH"],
        "close": [10.0, 1500.0, 10.5, 1520.0],
    })

    mgr = SharedArrowTableManager()
    mgr.register_dataframe("market_bars", df, broadcast_to_ray=False)

    catalog = DuckDBSharedCatalog(mgr)
    catalog.register_arrow_table("market_bars")

    # 1. SQL 零拷贝切片
    sliced_df = catalog.slice_market_bars(
        table_name="market_bars",
        symbols=["000001.SZ"],
        start_date="2025-01-02",
        end_date="2025-01-04",
    )
    assert len(sliced_df) == 2
    assert set(sliced_df["symbol"].unique()) == {"000001.SZ"}

    # 2. 复杂 SQL 聚合
    agg_table = catalog.query("SELECT symbol, AVG(close) as avg_px FROM market_bars GROUP BY symbol ORDER BY symbol")
    assert agg_table.num_rows == 2
    avg_series = agg_table.to_pandas().set_index("symbol")["avg_px"]
    assert abs(avg_series["000001.SZ"] - 10.25) < 1e-6
    assert abs(avg_series["600519.SH"] - 1510.0) < 1e-6
