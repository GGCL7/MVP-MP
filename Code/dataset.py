
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Batch as GeomBatch

from get_feature import smiles_to_data, extract_fingerprints


class MultiLabelMoleculeDataset(Dataset):


    def __init__(
        self,
        df_or_path,
        num_labels: int,
        smiles_col: str = "smiles",
        label_col: str = "label",
        cache: bool = True,
        labels_split: str = ",",
    ):
        super().__init__()


        if isinstance(df_or_path, str):
            df = pd.read_csv(df_or_path)
        else:
            df = df_or_path


        self.df = df.reset_index(drop=True)

        self.num_labels = num_labels
        self.smiles_col = smiles_col
        self.label_col = label_col
        self.labels_split = labels_split
        self.cache = cache


        self.smiles_list = self.df[self.smiles_col].tolist()
        self.labels_str_list = self.df[self.label_col].tolist()


        self.graph_cache = {} if cache else None
        self.fp_cache = {} if cache else None



    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        smiles = self.smiles_list[idx]
        label_str = self.labels_str_list[idx]


        if isinstance(label_str, str) and label_str.strip():

            label_indices = [
                int(i) for i in label_str.split(self.labels_split)
                if i.strip() != ""
            ]
            if len(label_indices) > 0:
                onehot = F.one_hot(
                    torch.tensor(label_indices, dtype=torch.long),
                    num_classes=self.num_labels
                ).sum(dim=0).float()  # [num_labels]
            else:
                onehot = torch.zeros(self.num_labels, dtype=torch.float32)
        else:
            onehot = torch.zeros(self.num_labels, dtype=torch.float32)


        if self.cache and idx in self.graph_cache:
            data_graph = self.graph_cache[idx]
        else:
            data_graph = smiles_to_data(smiles)   # PyG Data(x, edge_index, edge_attr)
            if self.cache:
                self.graph_cache[idx] = data_graph


        if self.cache and idx in self.fp_cache:
            fp = self.fp_cache[idx]
        else:
            fp = extract_fingerprints(smiles)     # torch.Tensor [fp_dim]
            if self.cache:
                self.fp_cache[idx] = fp

        return data_graph, fp, onehot


def collate_fn(batch):

    import torch

    graphs, fps, labels = zip(*batch)

    batch_graph = GeomBatch.from_data_list(graphs)
    fps = torch.stack(fps, dim=0)       # [B, fp_dim]
    labels = torch.stack(labels, dim=0) # [B, num_labels]

    return batch_graph, fps, labels





