# predict.py
# Usage example:
#   python predict.py \
#     --smiles "NC(=O)c1ccc[n+](C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(O)C3O)C(O)C2O)c1" \
#     --checkpoint best_model.pth \
#     --threshold 0.5

import argparse
import sys
from typing import Dict, Tuple

import torch
from torch_geometric.data import Batch as GeomBatch

from model import PathwayPredictor
from get_feature import smiles_to_data, extract_fingerprints


PATHWAY_FULL_NAMES: Dict[int, str] = {
    0: "Carbohydrate metabolism",
    1: "Energy metabolism",
    2: "Lipid metabolism",
    3: "Nucleotide metabolism",
    4: "Amino acid metabolism",
    5: "Metabolism of other amino acids",
    6: "Glycan metabolism",
    7: "Cofactors and vitamins",
    8: "Terpenoids and polyketides metabolism",
    9: "Biosynthesis of other secondary metabolites",
    10: "Xenobiotics biodegradation and metabolism",
}


def _load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt
    if isinstance(ckpt, dict):
        return {"model": ckpt}
    raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")


def _build_single_input(smiles: str, device: torch.device) -> Tuple[GeomBatch, torch.Tensor]:

    try:
        g = smiles_to_data(smiles)
        fp = extract_fingerprints(smiles)
    except Exception as e:
        raise ValueError(f"Failed to featurize SMILES. smiles={smiles}\nError: {e}") from e

    batch_graph = GeomBatch.from_data_list([g]).to(device)
    fps = fp.unsqueeze(0).to(device)
    return batch_graph, fps


def _init_model_from_single(
    batch_graph: GeomBatch,
    fps: torch.Tensor,
    num_labels: int,
    device: torch.device
) -> PathwayPredictor:
    node_in_dim = batch_graph.x.size(1)
    edge_in_dim = batch_graph.edge_attr.size(1) if getattr(batch_graph, "edge_attr", None) is not None else 0
    fp_dim = fps.size(1)

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
    return model


def main():
    parser = argparse.ArgumentParser(description="Predict KEGG metabolic pathway multi-labels for a given SMILES.")
    parser.add_argument("--smiles", type=str, required=True, help="Input SMILES string.")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Path to model checkpoint (.pth).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for multi-label prediction.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    num_labels = 11
    device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")

    batch_graph, fps = _build_single_input(args.smiles, device)

    model = _init_model_from_single(batch_graph, fps, num_labels=num_labels, device=device)

    ckpt = _load_checkpoint(args.checkpoint, device=device)
    model.load_state_dict(ckpt["model"], strict=True)
    best_epoch = ckpt.get("epoch", None)

    model.eval()
    with torch.no_grad():
        logits = model(batch_graph, fps)                 # [1, num_labels]
        probs = torch.sigmoid(logits).squeeze(0)         # [num_labels]


    print("========== Prediction ==========")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    if best_epoch is not None:
        print(f"Best epoch (from ckpt): {best_epoch}")
    print(f"SMILES: {args.smiles}")
    print(f"Threshold: {args.threshold:.3f}")
    print("================================\n")

    for i in range(num_labels):
        name = PATHWAY_FULL_NAMES.get(i, f"Label_{i}")
        p = float(probs[i].item())
        pred = 1 if p >= args.threshold else 0
        print(f"[{i:02d}] {name:<45s} prob={p:.4f}  pred={pred}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)
