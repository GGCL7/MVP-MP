import os
import ast
import json
import argparse
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import MultiLabelMoleculeDataset, collate_fn
from model import PathwayPredictor
from utils import set_seed, multilabel_metrics


def load_split_indices(index_path: str) -> Tuple[list, list, list]:
    """Read 3 lines: train / val / test indices (assumed 0-based)."""
    data_index = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data_index.append(ast.literal_eval(line))
    if len(data_index) != 3:
        raise ValueError("index file must contain exactly 3 non-empty lines: train/val/test index lists")
    return data_index[0], data_index[1], data_index[2]


def pathway_contrastive_loss(
    h: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    InfoNCE-style loss:
      positives for anchor i: samples sharing >=1 label with i (excluding itself).
    """
    device = h.device
    B = h.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=device)

    y = (labels > 0.5).float()  # [B, L]

    with torch.no_grad():
        inter = torch.matmul(y, y.t())  # [B, B]
        eye = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = (inter > 0) & (~eye)

    h_norm = F.normalize(h, dim=-1)
    sim = torch.matmul(h_norm, h_norm.t()) / temperature  # [B, B]

    losses = []
    for i in range(B):
        pos_idx = pos_mask[i].nonzero(as_tuple=False).view(-1)
        if pos_idx.numel() == 0:
            continue

        logits_i = sim[i].clone()
        logits_i[i] = -1e9  # mask self

        exp_logits = torch.exp(logits_i)
        denom = exp_logits.sum()
        num = exp_logits[pos_idx].sum()

        losses.append(-torch.log(num / (denom + eps) + eps))

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def batch_prototype_loss(
    h: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Build per-label prototypes from positives within the batch, then apply BCE on dot-product logits.
    Only labels with at least one positive in the current batch contribute to loss.
    """
    device = h.device
    B, d = h.size()
    if B == 0:
        return torch.tensor(0.0, device=device)

    y = labels.float()  # [B, L]
    pos_counts = y.sum(dim=0)  # [L]
    YT = y.t()  # [L, B]

    proto = torch.matmul(YT, h)  # [L, d]
    proto = proto / (pos_counts.unsqueeze(-1) + eps)

    logits_proto = torch.matmul(h, proto.t())  # [B, L]

    label_mask = (pos_counts > 0).float()  # [L]
    valid_label_cnt = label_mask.sum()
    if valid_label_cnt < 1.0:
        return torch.tensor(0.0, device=device)

    bce = nn.BCEWithLogitsLoss(reduction="none")
    per_entry = bce(logits_proto, y)  # [B, L]
    per_entry = per_entry * label_mask.unsqueeze(0)

    return per_entry.sum() / (valid_label_cnt * B)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    threshold: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    preds_all, trues_all = [], []

    for batch_graph, fps, labels in loader:
        batch_graph = batch_graph.to(device)
        fps = fps.to(device)
        labels = labels.to(device)

        logits = model(batch_graph, fps)  # [B, L]
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs

        preds_all.append(torch.sigmoid(logits))
        trues_all.append(labels)

    total_loss /= len(loader.dataset)
    preds_all = torch.cat(preds_all, dim=0)
    trues_all = torch.cat(trues_all, dim=0)

    metrics = multilabel_metrics(trues_all, preds_all, threshold=threshold)
    return total_loss, metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--index_path", type=str, required=True)

    p.add_argument("--num_labels", type=int, default=11)
    p.add_argument("--smiles_col", type=str, default="smiles")
    p.add_argument("--label_col", type=str, default="label")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lambda_contrast", type=float, default=0.5)
    p.add_argument("--lambda_proto", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.1)

    p.add_argument("--select_metric", type=str, default="f1", choices=["f1", "acc", "prec", "rec"])
    p.add_argument("--threshold", type=float, default=0.5)

    p.add_argument("--ckpt_path", type=str, default="best_model_by_val.pth")

    # Model dims/hparams (keep defaults aligned with your previous script)
    p.add_argument("--graph_hidden_dim", type=int, default=128)
    p.add_argument("--graph_out_dim", type=int, default=256)
    p.add_argument("--fp_hidden_dim", type=int, default=256)
    p.add_argument("--fusion_hidden_dim", type=int, default=256)
    p.add_argument("--final_dim", type=int, default=256)

    p.add_argument("--cache", action="store_true", help="Enable dataset caching if supported by your dataset class")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(args.csv_path)
    N = len(df)
    print(f"Total samples: {N}")

    train_idx, val_idx, test_idx = load_split_indices(args.index_path)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    all_idx = train_idx + val_idx + test_idx
    if len(all_idx) == 0:
        raise ValueError("Empty index lists.")
    if max(all_idx) >= N or min(all_idx) < 0:
        raise ValueError("Index out of bounds for the provided CSV.")

    dataset = MultiLabelMoleculeDataset(
        df_or_path=args.csv_path,
        num_labels=args.num_labels,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        cache=bool(args.cache),
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Infer input dimensions
    sample_graph, sample_fp, _ = dataset[0]
    node_in_dim = sample_graph.x.size(1)
    edge_in_dim = sample_graph.edge_attr.size(1)
    fp_dim = sample_fp.size(0)
    print(f"In dims: node={node_in_dim}, edge={edge_in_dim}, fp={fp_dim}")

    model = PathwayPredictor(
        num_labels=args.num_labels,
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        fp_dim=fp_dim,
        graph_hidden_dim=args.graph_hidden_dim,
        graph_out_dim=args.graph_out_dim,
        fp_hidden_dim=args.fp_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        final_dim=args.final_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_score = -1e9
    best_epoch = -1
    best_ckpt = None

    for epoch in range(1, args.epochs + 1):
        model.train()

        total = 0.0
        total_bce = 0.0
        total_con = 0.0
        total_proto = 0.0

        train_preds, train_trues = [], []

        for batch_graph, fps, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            batch_graph = batch_graph.to(device)
            fps = fps.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            h_c, logits = model(batch_graph, fps, return_emb=True)

            loss_bce = criterion(logits, labels)
            loss_con = pathway_contrastive_loss(h_c, labels, temperature=args.temperature)
            loss_proto = batch_prototype_loss(h_c, labels)

            loss = loss_bce + args.lambda_contrast * loss_con + args.lambda_proto * loss_proto
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            total += loss.item() * bs
            total_bce += loss_bce.item() * bs
            total_con += loss_con.item() * bs
            total_proto += loss_proto.item() * bs

            train_preds.append(torch.sigmoid(logits).detach())
            train_trues.append(labels.detach())

        total /= len(train_set)
        total_bce /= len(train_set)
        total_con /= len(train_set)
        total_proto /= len(train_set)

        train_preds = torch.cat(train_preds, dim=0)
        train_trues = torch.cat(train_trues, dim=0)
        train_metrics = multilabel_metrics(train_trues, train_preds, threshold=args.threshold)

        val_loss, val_metrics = evaluate(model, val_loader, device, criterion, threshold=args.threshold)

        print(
            f"[Epoch {epoch}] "
            f"TrainLoss={total:.4f} (BCE={total_bce:.4f}, Con={total_con:.4f}, Proto={total_proto:.4f}) | "
            f"Train Acc={train_metrics['acc']:.4f} Prec={train_metrics['prec']:.4f} "
            f"Rec={train_metrics['rec']:.4f} F1={train_metrics['f1']:.4f} | "
            f"ValLoss={val_loss:.4f} Val Acc={val_metrics['acc']:.4f} "
            f"Prec={val_metrics['prec']:.4f} Rec={val_metrics['rec']:.4f} F1={val_metrics['f1']:.4f}"
        )

        score = float(val_metrics[args.select_metric])
        if score > best_val_score:
            best_val_score = score
            best_epoch = epoch
            best_ckpt = {
                "model": model.state_dict(),
                "epoch": epoch,
                "best_val_score": best_val_score,
                "select_metric": args.select_metric,
                "val_metrics": {
                    "loss": val_loss,
                    "acc": float(val_metrics["acc"]),
                    "prec": float(val_metrics["prec"]),
                    "rec": float(val_metrics["rec"]),
                    "f1": float(val_metrics["f1"]),
                },
                "args": vars(args),
            }
            torch.save(best_ckpt, args.ckpt_path)
            print(f"Saved best checkpoint by Val {args.select_metric.upper()}={best_val_score:.4f} to {args.ckpt_path}")

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    test_loss, test_metrics = evaluate(model, test_loader, device, criterion, threshold=args.threshold)

    print("\n================= Final Summary =================")
    print(f"Best epoch by Val {ckpt.get('select_metric','?').upper()}: {ckpt.get('epoch','?')}")
    if "val_metrics" in ckpt:
        vm = ckpt["val_metrics"]
        print(f"Best Val  : Loss={vm['loss']:.4f} Acc={vm['acc']:.4f} Prec={vm['prec']:.4f} Rec={vm['rec']:.4f} F1={vm['f1']:.4f}")
    print(f"Final Test: Loss={test_loss:.4f} Acc={test_metrics['acc']:.4f} Prec={test_metrics['prec']:.4f} Rec={test_metrics['rec']:.4f} F1={test_metrics['f1']:.4f}")
    print("=================================================")

    # Optional: dump best config and metrics as JSON next to ckpt
    sidecar = os.path.splitext(args.ckpt_path)[0] + "_summary.json"
    summary = {
        "ckpt_path": args.ckpt_path,
        "best_epoch": int(ckpt.get("epoch", -1)),
        "select_metric": ckpt.get("select_metric", args.select_metric),
        "best_val_score": float(ckpt.get("best_val_score", best_val_score)),
        "best_val_metrics": ckpt.get("val_metrics", {}),
        "final_test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "final_test_loss": float(test_loss),
        "args": ckpt.get("args", vars(args)),
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved run summary to: {sidecar}")


if __name__ == "__main__":
    main()
