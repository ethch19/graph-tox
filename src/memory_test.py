import random
import sqlite3
import time
from pathlib import Path

import pandas as pd
from torch_geometric.loader import DataLoader

from src.data_prep import DrugDB

BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "drugs.db"


def test_memory():
    conn = sqlite3.connect(DB_DIR)

    print("Running query for inchi_keys...")
    query = "SELECT inchi_key FROM drugs WHERE smiles IS NOT NULL"
    keys_df = pd.read_sql_query(query, conn)
    conn.close()

    memory_bytes = keys_df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 * 1024)

    print("=" * 30)
    print(f"Total rows loaded: {len(keys_df):,}")
    print(f"Total RAM used by keys_df: {memory_mb:.2f} MB")
    print("=" * 30)


def test_getitem_speed():
    print("Initializing lazy-loader Dataset...")
    dataset = DrugDB(db_path=str(DB_DIR))

    total_items = len(dataset)
    test_indices = random.sample(range(total_items), 5)

    print(f"\nTesting 5 random molecules out of {total_items:,}...")

    for i, idx in enumerate(test_indices, 1):
        print(f"\n--- Test {i}/5 (Index: {idx}) ---")

        start_time = time.perf_counter()

        data = dataset[idx]

        fetch_time = time.perf_counter() - start_time

        if data is None:
            print(f"Time taken: {fetch_time:.5f} seconds")
            print("Result: None (Graph failed to parse, likely an invalid SMILES)")
        else:
            print(f"Time taken: {fetch_time:.5f} seconds")
            print(f"Graph Nodes (x) shape: {data.x.shape}")
            print(f"Graph Edges (edge_index) shape: {data.edge_index.shape}")
            print(f"LINCS assay shape: {data.lincs.shape}")
            print(f"ChEMBL assay shape: {data.chembl.shape}")


def get_mem_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def test_batch_memory():
    print(f"1. Baseline Python Memory: {get_mem_mb():.2f} MB")

    dataset = DrugDB(db_path=str(DB_DIR))
    print(f"2. Memory after loading keys_df: {get_mem_mb():.2f} MB")

    loader = DataLoader(dataset, batch_size=128, num_workers=0)

    print("\nFetching 50 batches (6,400 molecules) to check for leaks...")
    for i, batch in enumerate(loader, 1):
        if i % 10 == 0:
            print(f"   Batch {i:02d}/50 | Current Memory: {get_mem_mb():.2f} MB")
        if i == 50:
            break

    print("\nTest complete! If the memory stayed flat, you are ready for the HPC.")


if __name__ == "__main__":
    test_batch_memory()
