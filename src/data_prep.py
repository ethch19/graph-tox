import json
import os
import random
import sqlite3

import torch
from torch_geometric.data import Data, Dataset

from src.utils import smiles_to_graph


def get_worker_mem():
    """Returns current process RAM usage in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except:
        return 0.0


class DrugDB(Dataset):
    def __init__(self, db_path="../data/drugs.db"):
        super().__init__()
        self.db_path = db_path
        self.lincs_dim = 978
        self.chembl_dim = 1283

        conn = sqlite3.connect(self.db_path)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT inchi_key FROM drugs WHERE smiles IS NOT NULL")
        self.keys = [row[0] for row in cursor.fetchall()]
        conn.close()

        self.conn = None
        self.counter = 0
        print(f"Lazy-loaded {len(self.keys)} compound keys")

    def __len__(self):
        return len(self.keys)

    def _get_conn(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        return self.conn

    def __getitem__(self, idx):
        self.counter += 1
        if self.counter % 500 == 0:
            pid = os.getpid()
            print(
                f"[DEBUG] PID {pid} | Processed {self.counter} | RAM: {get_worker_mem():.2f} MB"
            )

        conn = self._get_conn()
        cursor = conn.cursor()

        while True:
            inchi_key = self.keys[idx]
            cursor.execute("SELECT smiles FROM drugs WHERE inchi_key = ?", (inchi_key,))
            row = cursor.fetchone()
            smiles = row[0] if row else None

            if smiles:
                cursor.execute(
                    """
                    SELECT sr.data_json 
                    FROM source_records sr 
                    JOIN sources s ON sr.source_id = s.id 
                    WHERE sr.drug_inchi_key = ? AND s.name = 'ChEMBL'
                """,
                    (inchi_key,),
                )
                chembl_row = cursor.fetchone()

                cursor.execute(
                    """
                    SELECT sr.data_json 
                    FROM source_records sr 
                    JOIN sources s ON sr.source_id = s.id 
                    WHERE sr.drug_inchi_key = ? AND s.name = 'LINCS_L1000_PhaseII'
                """,
                    (inchi_key,),
                )
                lincs_row = cursor.fetchone()

                x, edge_index, edge_attr = smiles_to_graph(smiles)
                if x is not None:
                    break

            idx = random.randint(0, len(self.keys) - 1)

            l_json = json.loads(lincs_row[0]) if lincs_row else {}
            c_json = json.loads(chembl_row[0]) if chembl_row else {}

            return Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                lincs=torch.tensor(
                    l_json.get("lincs_zscores_array", [0.0] * self.lincs_dim),
                    dtype=torch.float,
                ),
                chembl=torch.tensor(
                    c_json.get("chembl_pchembl_array", [0.0] * self.chembl_dim),
                    dtype=torch.float,
                ),
            )
