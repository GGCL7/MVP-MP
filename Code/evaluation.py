
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import MultiLabelMoleculeDataset, collate_fn
from model import PathwayPredictor
from utils import set_seed, multilabel_metrics


def load_split_indices(index_path: str):

    data_index = []
    with open(index_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            idx_list = ast.literal_eval(line)
            data_index.append(idx_list)

    assert len(data_index) == 3, "data_index.txt should contain：train / val / test"

    train_idx, val_idx, test_idx = data_index
    return train_idx, val_idx, test_idx


def main():
    csv_path = "/Users/guanjiahui/Desktop/Metabolic Pathway/MVML-MPI-main/data/kegg_dataset.csv"
    index_path = "/Users/guanjiahui/Desktop/Metabolic Pathway/MVML-MPI-main/data/data_index.txt"
    checkpoint_path = "best_model.pth"

    num_labels = 11
    batch_size = 64
    seed = 42

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(csv_path)
    N = len(df)
    print(f"总样本数: {N}")

    train_idx, val_idx, test_idx = load_split_indices(index_path)
    print(
        f"Train: {len(train_idx)}, Val: {len(val_idx)}, "
        f"Test: {len(test_idx)}"
    )


    full_dataset = MultiLabelMoleculeDataset(
        df_or_path=csv_path,
        num_labels=num_labels,
        smiles_col="smiles",
        label_col="label",
        cache=True,
    )

    test_dataset = Subset(full_dataset, test_idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )


    sample_graph, sample_fp, sample_y = full_dataset[0]
    node_in_dim = sample_graph.x.size(1)
    edge_in_dim = sample_graph.edge_attr.size(1)
    fp_dim = sample_fp.size(0)
    print(f"[Info] node_in_dim={node_in_dim}, edge_in_dim={edge_in_dim}, fp_dim={fp_dim}")


    model = PathwayPredictor(
        num_labels=num_labels,
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        fp_dim=fp_dim,
        graph_hidden_dim=128,
        graph_out_dim=256,
        fp_hidden_dim=256,
        fusion_hidden_dim=256,
        final_dim=256,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    best_epoch = ckpt.get("epoch", None)

    print("\n========== Loaded Checkpoint ==========")
    print(f"Loaded from: {checkpoint_path}")
    if best_epoch is not None:
        print(f"Best epoch in training: {best_epoch}")
    print("=======================================\n")


    criterion = nn.BCEWithLogitsLoss()

    model.eval()
    test_loss = 0.0
    test_preds, test_trues = [], []

    with torch.no_grad():
        for batch_graph, fps, labels in tqdm(test_loader, desc="[Eval Test]"):
            batch_graph = batch_graph.to(device)
            fps = fps.to(device)
            labels = labels.to(device)
            logits = model(batch_graph, fps)

            loss = criterion(logits, labels)
            test_loss += loss.item() * labels.size(0)

            test_preds.append(torch.sigmoid(logits))
            test_trues.append(labels)

    test_loss /= len(test_dataset)
    test_preds = torch.cat(test_preds, dim=0)
    test_trues = torch.cat(test_trues, dim=0)
    test_metrics = multilabel_metrics(test_trues, test_preds, threshold=0.5)

    print("=== Test Results (Loaded Best Model) ===")
    print(
        f"Acc={test_metrics['acc']:.4f}, "
        f"Prec={test_metrics['prec']:.4f}, "
        f"Rec={test_metrics['rec']:.4f}, "
        f"F1={test_metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
