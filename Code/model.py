
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_add
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils import degree, subgraph



class MVPool(nn.Module):

    def __init__(self,
                 in_channels: int,
                 ratio: float,
                 edge_attr_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio


        self.feat_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1)
        )


        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1)
        )

        self.view_weight = nn.Parameter(torch.ones(3))

    def _standardize(self, v: torch.Tensor) -> torch.Tensor:

        mean = v.mean()
        std = v.std()
        return (v - mean) / (std + 1e-6)

    def _diffusion_step(self,
                        x: torch.Tensor,
                        edge_index: torch.Tensor,
                        edge_attr: torch.Tensor) -> torch.Tensor:

        row, col = edge_index
        N = x.size(0)

        if edge_attr is not None:
            w_raw = self.edge_mlp(edge_attr).squeeze(-1)     # [E]
            edge_weight = F.softplus(w_raw) + 1e-6
        else:
            edge_weight = x.new_ones(row.size(0))

        deg = scatter_add(edge_weight, row, dim=0, dim_size=N)  # [N]
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]  # [E]


        x_diff = scatter_add(
            x[col] * norm.unsqueeze(-1),   # [E, F]
            row, dim=0, dim_size=N         # -> [N, F]
        )
        return x_diff


    def compute_scores(self,
                       x: torch.Tensor,
                       edge_index: torch.Tensor,
                       edge_attr: torch.Tensor,
                       batch: torch.Tensor = None) -> torch.Tensor:

        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        row, _ = edge_index
        N = x.size(0)


        deg_vec = degree(row, num_nodes=N, dtype=x.dtype)  # [N]

        struct_raw = torch.log(deg_vec + 1.0)              # [N]
        struct_score = self._standardize(struct_raw)       # [N]

        feat_raw = self.feat_mlp(x).squeeze(-1)            # [N]
        feat_score = self._standardize(feat_raw)           # [N]

        x_diff = self._diffusion_step(x, edge_index, edge_attr)  # [N, F]

        dot = (x * x_diff).sum(dim=-1)                     # [N]
        norm_x = x.norm(dim=-1) + 1e-6
        norm_xd = x_diff.norm(dim=-1) + 1e-6
        cos_sim = dot / (norm_x * norm_xd)                 # [-1,1]
        diff_raw = cos_sim
        diff_score = self._standardize(diff_raw)           # [N]

        score_mat = torch.stack(
            [struct_score, feat_score, diff_score],
            dim=-1
        )  # [N, 3]

        raw = score_mat @ self.view_weight                 # [N]
        score = torch.sigmoid(raw)                         # (0,1)
        return score


    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor):
        score = self.compute_scores(x, edge_index, edge_attr, batch)
        out_x, out_edge_index, out_edge_attr, out_batch = self._topk_pool(
            x, edge_index, edge_attr, batch, score
        )
        return out_x, out_edge_index, out_edge_attr, out_batch


    def _topk_pool(self,
                   x: torch.Tensor,
                   edge_index: torch.Tensor,
                   edge_attr: torch.Tensor,
                   batch: torch.Tensor,
                   score: torch.Tensor):
        num_nodes = x.size(0)
        device = x.device

        new_x_list, new_edge_index_list, new_edge_attr_list, new_batch_list = [], [], [], []

        for g in batch.unique():
            mask = (batch == g)
            idx = mask.nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue

            k = max(1, int(self.ratio * idx.numel()))
            g_scores = score[idx]
            perm_local = torch.topk(g_scores, k, sorted=False).indices
            keep = idx[perm_local]

            node_id_map = {old.item(): i for i, old in enumerate(keep)}
            node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            node_mask[keep] = True

            e_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            e_idx = edge_index[:, e_mask]
            e_attr = edge_attr[e_mask]

            if e_idx.numel() > 0:
                new_e0 = torch.tensor(
                    [node_id_map[int(i)] for i in e_idx[0].cpu().tolist()],
                    dtype=torch.long, device=device
                )
                new_e1 = torch.tensor(
                    [node_id_map[int(i)] for i in e_idx[1].cpu().tolist()],
                    dtype=torch.long, device=device
                )
                new_e = torch.stack([new_e0, new_e1], dim=0)
            else:
                new_e = torch.empty((2, 0), dtype=torch.long, device=device)
                e_attr = torch.empty(
                    (0, edge_attr.size(-1)),
                    dtype=edge_attr.dtype,
                    device=device
                )

            new_x = x[keep] * score[keep].view(-1, 1)
            new_batch = torch.full(
                (keep.size(0),),
                g.item(), dtype=batch.dtype, device=device
            )

            new_x_list.append(new_x)
            new_edge_index_list.append(new_e)
            new_edge_attr_list.append(e_attr)
            new_batch_list.append(new_batch)

        out_x = torch.cat(new_x_list, dim=0)
        out_edge_index = torch.cat(new_edge_index_list, dim=1)
        out_edge_attr = torch.cat(new_edge_attr_list, dim=0)
        out_batch = torch.cat(new_batch_list, dim=0)

        return out_x, out_edge_index, out_edge_attr, out_batch


class GINGraphEncoder(nn.Module):


    def __init__(self,
                 node_in_dim: int,
                 edge_in_dim: int,
                 hidden_dim: int = 128,
                 out_dim: int = 256,
                 pool_ratio: float = 0.5,
                 dropout: float = 0.2):
        super().__init__()

        self.pool_ratio = pool_ratio
        self.hidden_dim = hidden_dim

        nn1 = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = GINEConv(nn1, edge_dim=edge_in_dim)

        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv2 = GINEConv(nn2, edge_dim=edge_in_dim)

        self.pool = MVPool(
            in_channels=hidden_dim,
            ratio=pool_ratio,
            edge_attr_dim=edge_in_dim,
        )

        self.sub_readout_proj = nn.Linear(hidden_dim, hidden_dim)
        self.back_proj = nn.Linear(hidden_dim, hidden_dim)
        self.back_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    @torch.no_grad()
    def _topk_per_graph(self,
                        score: torch.Tensor,
                        batch_ids: torch.Tensor,
                        ratio: float):

        device = score.device
        num_graphs = int(batch_ids.max().item()) + 1
        idx_list = []
        for g in range(num_graphs):
            mask = (batch_ids == g)
            n_g = int(mask.sum().item())
            if n_g == 0:
                continue
            k_g = max(1, int(math.ceil(ratio * n_g)))
            s_g = score[mask]                              # [n_g]
            _, topk_loc = torch.topk(s_g, k_g, sorted=False)
            global_idx = mask.nonzero(as_tuple=False).view(-1)[topk_loc]
            idx_list.append(global_idx)
        if len(idx_list) == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.sort(torch.cat(idx_list, dim=0))[0]

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        h1 = self.conv1(x, edge_index, edge_attr)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)   # [N, H]

        scores = self.pool.compute_scores(h1, edge_index, edge_attr, batch)  # [N]
        idx_keep = self._topk_per_graph(scores, batch, self.pool_ratio)      # [K]

        sub_edge_index, sub_edge_attr = subgraph(
            idx_keep,
            edge_index,
            edge_attr,
            relabel_nodes=True,
            num_nodes=h1.size(0)
        )
        h1_sub = h1[idx_keep]              # [K, H]
        sub_batch = batch[idx_keep]        # [K]

        h2_sub = self.conv2(h1_sub, sub_edge_index, sub_edge_attr)
        h2_sub = F.relu(h2_sub)
        h2_sub = self.dropout(h2_sub)

        hs = global_mean_pool(h2_sub, sub_batch)    # [B, H]
        hs_proj = self.sub_readout_proj(hs)         # [B, H]
        hs_broadcast = hs_proj[batch]               # [N, H]

        gate = self.back_gate(torch.cat([h1, hs_broadcast], dim=-1))  # [N, H]
        h_refined = h1 + gate * self.back_proj(hs_broadcast)          # [N, H]

        g = global_mean_pool(h_refined, batch)      # [B, H]
        g = self.out_proj(g)                        # [B, out_dim]
        g = F.relu(g)
        return g

class FingerprintEncoder(nn.Module):
    def __init__(self,
                 fp_dim: int,
                 hidden_dim: int = 256,
                 out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, fp):
        return self.mlp(fp)

class BiGateFusion(nn.Module):
    def __init__(self,
                 in_dim_graph: int,
                 in_dim_fp: int,
                 hidden_dim: int = 256,
                 out_dim: int = 256):
        super().__init__()
        self.proj_g = nn.Linear(in_dim_graph, hidden_dim)
        self.proj_f = nn.Linear(in_dim_fp, hidden_dim)

        self.gate_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.out_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, h_g, h_fp):
        g = self.proj_g(h_g)
        f = self.proj_f(h_fp)

        concat = torch.cat([g, f], dim=-1)
        gate = torch.sigmoid(self.gate_net(concat))      # [B, hidden_dim]

        h_mix = gate * g + (1.0 - gate) * f
        h_bi = g * f
        fused = torch.cat([h_mix, h_bi], dim=-1)         # [B, 2*hidden_dim]

        out = self.out_net(fused)                        # [B, out_dim]
        return out

class PathwayPredictor(nn.Module):
    def __init__(self,
                 num_labels: int,
                 node_in_dim: int,
                 edge_in_dim: int,
                 fp_dim: int,
                 graph_hidden_dim: int = 128,
                 graph_out_dim: int = 256,
                 fp_hidden_dim: int = 256,
                 fusion_hidden_dim: int = 256,
                 final_dim: int = 256):
        super().__init__()

        self.graph_encoder = GINGraphEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=graph_hidden_dim,
            out_dim=graph_out_dim,
            pool_ratio=0.5,
        )

        self.fp_encoder = FingerprintEncoder(
            fp_dim=fp_dim,
            hidden_dim=fp_hidden_dim,
            out_dim=graph_out_dim,
        )

        self.fusion = BiGateFusion(
            in_dim_graph=graph_out_dim,
            in_dim_fp=graph_out_dim,
            hidden_dim=fusion_hidden_dim,
            out_dim=final_dim,
        )

        self.classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(final_dim, num_labels),
        )

    def forward(self, batch_graph, fps, return_emb: bool = False):

        h_g = self.graph_encoder(batch_graph)    # [B, d]
        h_fp = self.fp_encoder(fps)              # [B, d]
        h_c = self.fusion(h_g, h_fp)             # [B, final_dim]
        logits = self.classifier(h_c)            # [B, num_labels]

        if return_emb:
            return h_c, logits
        return logits

