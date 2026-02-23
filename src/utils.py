import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC, BT.DATIVE]
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
