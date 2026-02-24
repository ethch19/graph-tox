import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


def smiles_to_graph(smiles):
    """
    Converts a SMILES string into the exact PyTorch Geometric Data
    format expected by the MolCLR pretrained weights.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    # Node Features (Atoms): [AtomicNum Index, Chirality Index]
    type_idx, chirality_idx = [], []
    for atom in mol.GetAtoms():
        try:
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        except ValueError:
            return None, None, None

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    x = torch.cat([x1, x2], dim=-1)  # Shape: [NumAtoms, 2]

    # Edge Features (Bonds): [BondType Index, BondDirection Index]
    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]  # Graphs are bidirectional

        try:
            feat = [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir()),
            ]
        except ValueError:
            return None, None, None

        edge_feat.append(feat)
        edge_feat.append(feat)  # Add twice for bidirectional

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(edge_feat, dtype=torch.long)

    return x, edge_index, edge_attr


def generate_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(
    data_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1
):  # Murcko scaffold split
    scaffolds = {}
    for i, data in enumerate(data_list):
        scaffold = generate_scaffold(data.smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = []
        scaffolds[scaffold].append(i)

    scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]

    train_idx, valid_idx, test_idx = [], [], []
    train_cutoff = frac_train * len(data_list)
    valid_cutoff = (frac_train + frac_valid) * len(data_list)

    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    return (
        [data_list[i] for i in train_idx],
        [data_list[i] for i in valid_idx],
        [data_list[i] for i in test_idx],
    )


def cal_roc_auc(all_preds, all_targets, num_tasks):
    roc_aucs = []
    for i in range(num_tasks):
        valid_idx = ~np.isnan(all_targets[:, i])
        valid_targets = all_targets[valid_idx, i]
        valid_preds = all_preds[valid_idx, i]

        if len(np.unique(valid_targets)) == 2:
            roc_aucs.append(roc_auc_score(valid_targets, valid_preds))

    return np.mean(roc_aucs) if roc_aucs else 0.0
