import torch
import torch.nn as nn
from lib.utils import get_shortpath_num
from MPformer.MPformer import MPformer


class SpaAttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value, spatial_attn_bias=None):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)  # (num_heads * batch_size, ..., head_dim, src_length)
        attn_score = (query @ key) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if spatial_attn_bias is not None:
            attn_score = attn_score + spatial_attn_bias.view(1, 1, tgt_length, src_length)

        if self.mask:
            mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class TemAttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads, kernel, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Conv2d(model_dim, model_dim, kernel_size=(1, kernel), stride=(1, 1), padding=(0, int(kernel / 2)))
        self.FC_K = nn.Conv2d(model_dim, model_dim, kernel_size=(1, kernel), stride=(1, 1), padding=(0, int(kernel / 2)))
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        key = self.FC_K(key.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)  # (num_heads * batch_size, ..., head_dim, src_length)
        attn_score = (query @ key) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SpaTransformerLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim, num_heads, dropout, mask=False):
        super().__init__()

        self.attn = SpaAttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2, spatial_attn_bias=None):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x, spatial_attn_bias)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class TemTransformerLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim, num_heads, kernel, dropout, mask=False):
        super().__init__()

        self.attn = TemAttentionLayer(model_dim, num_heads, kernel, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class TSAformer(nn.Module):

    def __init__(
        self,
        dataset,
        num_nodes,
        in_steps=12,
        out_steps=12,
        input_dim=1,
        output_dim=1,
        input_embedding_dim=32,
        tod_embedding_dim=32,
        dow_embedding_dim=32,
        spatial_embedding_dim=32,
        adaptive_embedding_dim=128,
        long_history_embedding_dim=128,
        feed_forward_dim=512,
        num_heads=8,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = input_embedding_dim + tod_embedding_dim + dow_embedding_dim + spatial_embedding_dim + adaptive_embedding_dim

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(288, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            adj_matrix = torch.load(f"../data/{dataset}/adj_matrix.pt")
            self.degree_matrix = adj_matrix.sum(dim=1)
            self.shortest_path_matrix = get_shortpath_num(adj_matrix)
            self.spatial_embedding = nn.Embedding(self.degree_matrix.max().item() + 1, spatial_embedding_dim)
            self.spatial_attn_bias_embedding = nn.Embedding(self.shortest_path_matrix.max().item() + 1, 1)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim)))

        self.attn_layers_tem = nn.ModuleList(
            [
                TemTransformerLayer(self.model_dim, feed_forward_dim, num_heads, 1, dropout),
                TemTransformerLayer(self.model_dim, feed_forward_dim, num_heads, 3, dropout),
                TemTransformerLayer(self.model_dim, feed_forward_dim, num_heads, 5, dropout),
            ]
        )
        self.attn_layers_spa = nn.ModuleList([SpaTransformerLayer(self.model_dim, feed_forward_dim, num_heads, dropout) for _ in range(num_layers)])

        self.hidden_states_proj = nn.Sequential(
            nn.Linear(long_history_embedding_dim, feed_forward_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(feed_forward_dim, self.in_steps * self.model_dim),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(self.in_steps * self.model_dim, feed_forward_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(feed_forward_dim, self.out_steps * self.output_dim),
        )

    def forward(self, x, hidden_states):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]
        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]

        if self.tod_embedding_dim > 0:  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            tod_emb = self.tod_embedding(tod.long())
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            dow_emb = self.dow_embedding(dow.long())
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:  # (batch_size, in_steps, num_nodes, spatial_embedding_dim)
            degree = self.degree_matrix.expand(size=(batch_size, self.in_steps, self.num_nodes))
            spa_emb = self.spatial_embedding(degree.to(x.device))
            features.append(spa_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(size=(batch_size, *self.adaptive_embedding.shape))
            features.append(adp_emb)
        
        x = torch.cat(features, dim=-1)

        for attn in self.attn_layers_tem:
            x = attn(x, dim=1)

        self.shortest_path_matrix = self.shortest_path_matrix.to(x.device)
        spatial_attn_bias = self.spatial_attn_bias_embedding(self.shortest_path_matrix)
        for attn in self.attn_layers_spa:
            x = attn(x, dim=2, spatial_attn_bias=spatial_attn_bias)

        out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
        out = out.reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)

        hidden_states = self.hidden_states_proj(hidden_states)
        out = out + hidden_states

        out = self.output_proj(out)
        out = out.view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
        out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)

        return out


class MPTSAformer(nn.Module):
    def __init__(self, dataset, device, TSAformer_args, pretrain_length, mask_ratio):
        super().__init__()

        self.dataset = dataset
        self.pretrain_length = pretrain_length
        self.mask_ratio = mask_ratio
        self.device = device

        self.MPformer = MPformer(mode="forecasting", num_nodes=TSAformer_args["num_nodes"], mask_ratio=self.mask_ratio, in_steps=self.pretrain_length)
        self.load_model()

        self.TSAformer = TSAformer(dataset, **TSAformer_args)

    def load_model(self):
        checkpoint_dict = torch.load(f"../pretrain_model/{self.dataset}/MPformer-{self.dataset}-{self.pretrain_length}-{self.mask_ratio}.pt", map_location=self.device)
        self.MPformer.load_state_dict(checkpoint_dict)
        for param in self.MPformer.parameters():
            param.requires_grad = False

    def forward(self, x, long_history_data):
        hidden_states = self.MPformer(long_history_data)
        hidden_states = hidden_states[..., -1, :]

        return self.TSAformer(x, hidden_states)

