import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import add_self_loops, softmax
import math

# GAGB
class GAGB(MessagePassing):
    def __init__(self, in_channels, out_channels, pos_dim):
        super().__init__(aggr='mean')
        self.lin_self = Linear(in_channels, out_channels)
        self.lin_neigh = Linear(in_channels, out_channels)
        self.lin_q = Linear(in_channels, out_channels)  # For attention Q
        self.lin_k = Linear(in_channels, out_channels)  # For K

        # Input here is 2 * pos_dim because it's concatenation of pos_i - pos_j and pos_j - pos_i
        self.bias_proj = Linear(2 * pos_dim, 1)
        self.temp = nn.Parameter(torch.tensor(1.0))

        glorot(self.lin_self.weight)
        glorot(self.lin_neigh.weight)
        glorot(self.lin_q.weight)
        glorot(self.lin_k.weight)
        glorot(self.bias_proj.weight)
        self.pos_dim = pos_dim

    def forward(self, x, edge_index, edge_weight=None, pos=None):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if edge_weight is not None:
            self_loop_weight = torch.ones(x.size(0), device=x.device)
            edge_weight = torch.cat([edge_weight, self_loop_weight], dim=0)

        # pos needs to be passed to propagate
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, pos=pos)

    def message(self, x_i, x_j, edge_weight, pos_i, pos_j, index, ptr, size_i):
        # 1. Calculate base Attention Score
        q = self.lin_q(x_i)
        k = self.lin_k(x_j)
        score = (q * k).sum(dim=-1) / math.sqrt(k.size(-1))

        # 2. Calculate relative position Bias (Geometry Awareness)
        B = torch.zeros_like(score)
        if pos_i is not None and pos_j is not None:
            # This approach explicitly captures "where node i is relative to node j/distance"
            rel_pos = torch.cat([pos_i - pos_j, pos_j - pos_i], dim=-1)
            B = self.bias_proj(rel_pos).squeeze(-1)

        # Handle self-loop bias
        mask_self = (index == torch.arange(x_j.size(0), device=index.device))
        B[mask_self] = 0

        # 3. Fuse Score + Bias
        attn = softmax((score + B) / self.temp, index, ptr, size_i)

        # 4. Message weighting
        msg = x_j * attn.unsqueeze(-1)
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)

        return msg

    def update(self, aggr_out, x):
        return self.lin_self(x) + self.lin_neigh(aggr_out)


# SGB
class SGB(nn.Module):
    def __init__(self, input_dim, num_nodes, alpha=3.0):
        super().__init__()
        self.alpha = alpha
        self.weight1 = nn.Parameter(torch.randn(num_nodes, input_dim))
        self.weight2 = nn.Parameter(torch.randn(num_nodes, input_dim))
        self.act = nn.ReLU()

    def forward(self):
        adj = torch.mm(self.weight1, self.weight2.T)
        adj = self.act(adj)
        adj = F.softmax(adj * self.alpha, dim=-1)
        return adj


# DATM
class DATM(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1, kernel_size=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.conv_ffn = nn.Sequential(
            # 1. Dimension expansion
            nn.Linear(dim, dim * 4),
            nn.GELU(),

            # 2. Depthwise Conv
            RearrangeLayer(),
            nn.Conv1d(dim * 4, dim * 4, kernel_size=kernel_size, groups=dim * 4, padding=kernel_size // 2),
            nn.GELU(),

            # 3. Dimension reduction
            RestoreLayer(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. Global Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)

        # 2. Local Convolution FFN
        x_norm2 = self.norm2(x)
        conv_out = self.conv_ffn(x_norm2)
        x = x + conv_out

        return x


# Helper layer for dimension adjustment in Sequential
class RearrangeLayer(nn.Module):
    def forward(self, x): return x.transpose(1, 2)  # [B, T, D] -> [B, D, T]

class RestoreLayer(nn.Module):
    def forward(self, x): return x.transpose(1, 2)  # [B, D, T] -> [B, T, D]

# Gating layer
class GatedFusionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, 1),
            nn.Sigmoid()
        )
        # Initialize to 0.5
        self.last_static_weight = 0.5

    def forward(self, h_static, h_dynamic):
        combined = torch.cat([h_static, h_dynamic], dim=-1)

        z = self.gate_net(combined)

        self.last_static_weight = z.detach().mean().item()

        return z * h_static + (1 - z) * h_dynamic


# --- 3. Main Model: GeoDSTG ---
class GeoDSTG(nn.Module):
    def __init__(self, history_steps, out_channel=1, hidden_size=64,
                 num_heads=4, dropout=0.3, num_nodes=6):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_size = hidden_size

        self.input_proj = nn.Linear(1, hidden_size)
        self.pos_emb_time = nn.Parameter(torch.randn(1, history_steps, hidden_size))
        self.pos_emb_node = nn.Parameter(torch.randn(1, num_nodes, hidden_size))

        # Dynamic graph learner
        self.graph_learner = SGB(input_dim=64, num_nodes=num_nodes)

        self.temporal_layer1 = DATM(hidden_size, num_heads, dropout)
        self.temporal_layer2 = DATM(hidden_size, num_heads, dropout)

        # Spatial module
        self.spatial_static1 = GAGB(hidden_size, hidden_size, pos_dim=hidden_size)
        self.spatial_static2 = GAGB(hidden_size, hidden_size, pos_dim=hidden_size)

        self.dynamic_fc1 = nn.Linear(hidden_size, hidden_size)
        self.dynamic_fc2 = nn.Linear(hidden_size, hidden_size)

        # Gated fusion
        self.gate1 = GatedFusionLayer(hidden_size)
        self.gate2 = GatedFusionLayer(hidden_size)

        self.norm_s1 = nn.LayerNorm(hidden_size)
        self.norm_s2 = nn.LayerNorm(hidden_size)

        # Output module
        self.sensor_attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
            nn.Softmax(dim=1)
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, out_channel)
        )

    def forward(self, data):
        # Note: Here we take an additional edge_attr (i.e., edge_weight)
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        batch_size = batch.max().item() + 1

        # 1. Embedding and position encoding
        x = self.input_proj(x.unsqueeze(-1)) + self.pos_emb_time.to(x.device)

        # 2. First round of temporal processing
        x = self.temporal_layer1(x)
        x_spatial_in = x.mean(dim=1)  # [B*N, H]

        # 3. Prepare for spatial interaction
        x_node_view = x_spatial_in.view(batch_size, self.num_nodes, self.hidden_size)
        x_node_view = x_node_view + self.pos_emb_node.to(x.device)

        adj_dynamic = self.graph_learner()
        x_flat = x_node_view.view(-1, self.hidden_size)

        # --- 4. First round: Dual-stream spatial interaction ---

        # Branch A: Static graph (KNN/Pearson) - using WeightedSAGE
        h_static = self.spatial_static1(x_flat, edge_index, edge_weight)

        # Branch B: Dynamic graph
        h_dyn = self.dynamic_fc1(x_node_view)
        h_dyn = torch.matmul(adj_dynamic.unsqueeze(0), h_dyn)
        h_dyn_flat = h_dyn.view(-1, self.hidden_size)

        # Gated fusion
        h_fused = self.gate1(h_static, h_dyn_flat)
        h_s1 = self.norm_s1(h_fused + x_flat)

        # --- 5. Second round: Spatio-temporal interleaving ---
        x_t2 = x + h_s1.unsqueeze(1)
        x_t2 = self.temporal_layer2(x_t2)
        x_spatial_in2 = x_t2.mean(dim=1)

        x_node_view2 = x_spatial_in2.view(batch_size, self.num_nodes, self.hidden_size)
        x_flat2 = x_spatial_in2

        # --- 6. Second round: Dual-stream interaction ---

        # Branch A: Static graph - using WeightedSAGE
        h_static2 = self.spatial_static2(x_flat2, edge_index, edge_weight)

        # Branch B: Dynamic graph
        h_dyn2 = self.dynamic_fc2(x_node_view2)
        h_dyn2 = torch.matmul(adj_dynamic.unsqueeze(0), h_dyn2)
        h_dyn2_flat = h_dyn2.view(-1, self.hidden_size)

        # Gated fusion
        h_fused2 = self.gate2(h_static2, h_dyn2_flat)
        h_s2 = self.norm_s2(h_fused2 + x_flat2)

        # --- 7. Output aggregation ---
        final_node_repr = h_s2.view(batch_size, self.num_nodes, self.hidden_size)
        attn_weights = self.sensor_attention(final_node_repr)
        graph_repr = (final_node_repr * attn_weights).sum(dim=1)

        return self.fc_out(graph_repr)

