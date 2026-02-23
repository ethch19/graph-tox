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
        return len(self.keys_df)

    def __getitem__(self, idx):
        inchi_key = self.keys_df.iloc[idx]["inchi_key"]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT smiles FROM drugs WHERE inchi_key = ?", (inchi_key,))
        smiles_row = cursor.fetchone()
        smiles = smiles_row[0] if smiles_row else None

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
        chembl_json_str = chembl_row[0] if chembl_row else "{}"

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
        lincs_json_str = lincs_row[0] if lincs_row else "{}"

        conn.close()

        if smiles is None:
            return None

        x, edge_index, edge_attr = smiles_to_graph(smiles)
        if x is None:
            return None

        lincs_data = json.loads(lincs_json_str)
        chembl_data = json.loads(chembl_json_str)

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
