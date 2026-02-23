import json
import sqlite3

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from src.utils import smiles_to_graph


class DrugDB(Dataset):
    def __init__(self, db_path="../data/drugs.db"):
        super().__init__()
        self.db_path = db_path
        self.lincs_dim = 978
        self.chembl_dim = 1283

        conn = sqlite3.connect(self.db_path)

        query = "SELECT inchi_key FROM drugs WHERE smiles IS NOT NULL"
        self.keys_df = pd.read_sql_query(query, conn)
        conn.close()

        print(f"Lazy-loaded {len(self.keys_df)} compound keys")

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        inchi_key = self.keys_df.iloc[idx]["inchi_key"]

        query = f"""
            SELECT 
                d.smiles,
                sr_chembl.data_json as chembl_json,
                sr_lincs.data_json as lincs_json
            FROM drugs d
            LEFT JOIN source_records sr_chembl ON d.inchi_key = sr_chembl.drug_inchi_key
            LEFT JOIN sources s_chembl ON sr_chembl.source_id = s_chembl.id AND s_chembl.name = 'ChEMBL'
            LEFT JOIN source_records sr_lincs ON d.inchi_key = sr_lincs.drug_inchi_key
            LEFT JOIN sources s_lincs ON sr_lincs.source_id = s_lincs.id AND s_lincs.name = 'LINCS_L1000_PhaseII'
            WHERE d.inchi_key = '{inchi_key}'
        """

        conn = sqlite3.connect(self.db_path)
        row = pd.read_sql_query(query, conn).iloc[0]
        conn.close()

        x, edge_index, edge_attr = smiles_to_graph(row["smiles"])
        if x is None:
            return None

        lincs_data = (
            json.loads(row["lincs_json"]) if pd.notnull(row["lincs_json"]) else {}
        )
        chembl_data = (
            json.loads(row["chembl_json"]) if pd.notnull(row["chembl_json"]) else {}
        )

        lincs_array = lincs_data.get("lincs_zscores_array", [])
        chembl_array = chembl_data.get("chembl_pchembl_array", [])

        if len(lincs_array) != self.lincs_dim:
            lincs_array = [0.0] * self.lincs_dim
        if len(chembl_array) != self.chembl_dim:
            chembl_array = [0.0] * self.chembl_dim

        lincs_tensor = torch.tensor(lincs_array, dtype=torch.float)
        chembl_tensor = torch.tensor(chembl_array, dtype=torch.float)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            lincs=lincs_tensor,
            chembl=chembl_tensor,
        )
