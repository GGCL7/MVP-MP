import numpy as np
import rdkit
import torch
from rdkit import Chem
from rdkit.Chem import Mol, Atom, Bond
from torch_geometric.data import Data
from typing import List
import networkx as nx
import matplotlib.pyplot as plt
import random

atom_types = ['H', 'C', 'B', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
formal_charges = [-1, -2, 1, 2, 0]
degree = [0, 1, 2, 3, 4, 5, 6]
num_hs = [0, 1, 2, 3, 4]
local_chiral_tags = [0, 1, 2, 3]
hybridization = [
    rdkit.Chem.rdchem.HybridizationType.S,
    rdkit.Chem.rdchem.HybridizationType.SP,
    rdkit.Chem.rdchem.HybridizationType.SP2,
    rdkit.Chem.rdchem.HybridizationType.SP3,
    rdkit.Chem.rdchem.HybridizationType.SP3D,
    rdkit.Chem.rdchem.HybridizationType.SP3D2,
    rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
]
bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}
def one_hot_embedding(value: int, options: List[int]) -> List[int]:
    embedding = [0] * (len(options) + 1)
    index = options.index(value) if value in options else -1
    embedding[index] = 1
    return embedding

def get_node_features(atoms: List[Atom]) -> np.ndarray:
    num_features = (len(atom_types) + 1) + \
                   (len(degree) + 1) + \
                   (len(formal_charges) + 1) + \
                   (len(num_hs) + 1) + \
                   (len(hybridization) + 1) + \
                   2  # 43

    node_features = np.zeros((len(atoms), num_features))
    for node_index, node in enumerate(atoms):
        features = one_hot_embedding(node.GetSymbol(), atom_types)  # Atom symbol
        features += one_hot_embedding(node.GetTotalDegree(), degree)  # Number of bonds
        features += one_hot_embedding(node.GetFormalCharge(), formal_charges)  # Formal charge
        features += one_hot_embedding(node.GetTotalNumHs(), num_hs)  # Number of hydrogens
        features += one_hot_embedding(node.GetHybridization(), hybridization)  # Hybridization state
        features += [int(node.GetIsAromatic())]  # Aromaticity
        features += [node.GetMass() * 0.01]  # Atomic mass / 100
        node_features[node_index, :] = features

    return np.array(node_features, dtype=np.float32)

def get_edge_features(bonds: List[Bond]) -> np.ndarray:
    num_features = (len(bond_types) + 1) + 2  # 7

    edge_features = np.zeros((len(bonds) * 2, num_features))
    for edge_index, edge in enumerate(bonds):
        features = one_hot_embedding(str(edge.GetBondType()), bond_types)  # Bond type
        features += [int(edge.GetIsConjugated())]  # Conjugation
        features += [int(edge.IsInRing())]  # Ring membership

        # Encode both directed edges to get undirected edge
        edge_features[2 * edge_index: 2 * edge_index + 2, :] = features

    return np.array(edge_features, dtype=np.float32)
def adjacency_to_undirected_edge_index(adj: np.ndarray) -> np.ndarray:
    adj = np.triu(np.array(adj, dtype=int))  # Keep upper triangular matrix
    array_adj = np.array(np.nonzero(adj), dtype=int)  # Non-zero indices in adjacency matrix
    edge_index = np.zeros((2, 2 * array_adj.shape[1]), dtype=int)  # Placeholder for edges
    edge_index[:, ::2] = array_adj
    edge_index[:, 1::2] = np.flipud(array_adj)
    return edge_index

def mol_to_data(mol: Mol) -> Data:
    # Generate edge index (adjacency matrix)
    adj = Chem.GetAdjacencyMatrix(mol)
    edge_index = adjacency_to_undirected_edge_index(adj)

    # Generate edge features
    bonds = [mol.GetBondBetweenAtoms(int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.shape[1] // 2)]
    edge_features = get_edge_features(bonds)

    # Generate node features
    atoms = mol.GetAtoms()
    node_features = get_node_features(atoms)

    # Create Data object
    data = Data(x=torch.tensor(node_features, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_features, dtype=torch.float))

    return data

def smiles_to_3d_mol(smiles: str) -> Mol:
    mol = Chem.MolFromSmiles(smiles)
    return mol


