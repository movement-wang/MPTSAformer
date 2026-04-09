import torch
from torch import nn

from .patch import PatchEmbedding
from .mask_generator import MaskGenerator
from .transformer_layers import TransformerLayer


class MPformer(nn.Module):

    def __init__(
        self,
        mode,
        num_nodes,
        mask_ratio,
        in_steps,
        encoder_depth=4,
        input_dim=1,
        patch_size=12,
        embedding_dim=128,
        tod_embedding_dim=128,
        dow_embedding_dim=128,
        adaptive_embedding_dim=128,
        feed_forward_dim=256,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.mode = mode
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(288, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(nn.Parameter(torch.empty(num_nodes, in_steps // patch_size, adaptive_embedding_dim)))

        # encoder
        self.patch_embedding = PatchEmbedding(patch_size, input_dim, embedding_dim)
        self.encoder = nn.ModuleList([TransformerLayer(self.embedding_dim, feed_forward_dim, num_heads, dropout) for _ in range(encoder_depth)])

        # decoder
        self.enc_2_dec = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.mask_token = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, 1, 1, self.embedding_dim)), std=0.02)
        self.decoder = TransformerLayer(self.embedding_dim, feed_forward_dim, num_heads, dropout)

        # prediction (reconstruction) layer
        self.output_layer = nn.Linear(self.embedding_dim, patch_size)

    def encoding(self, long_term_history, mask=True):
        long_term_history = long_term_history.permute(0, 2, 1, 3)
        batch_size, num_node, len_time_series, num_feat = long_term_history.shape

        patches = self.patch_embedding(long_term_history[..., :1])  # B, N, d, P
        patches = patches.permute(0, 1, 3, 2)  # B, N, P, d

        patch_index = [i for i in range(0, len_time_series, self.patch_size)]
        if self.tod_embedding_dim > 0:
            tod = long_term_history[..., patch_index, 1]
            self.tod = self.tod_embedding(tod.long())
            patches = patches + self.tod
        if self.dow_embedding_dim > 0:
            dow = long_term_history[..., patch_index, 2]
            self.dow = self.dow_embedding(dow.long())
            patches = patches + self.dow
        if self.adaptive_embedding_dim > 0:
            self.adp = self.adaptive_embedding.expand(size=(batch_size, *self.adaptive_embedding.shape))
            patches = patches + self.adp

        if mask:
            Maskg = MaskGenerator(patches.shape[2], self.mask_ratio)
            unmasked_token_index, masked_token_index = Maskg.uniform_rand()
            encoder_input = patches[..., unmasked_token_index, :]
            for layer in self.encoder:
                encoder_input = layer(encoder_input)
            hidden_states_unmasked = encoder_input
        else:
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches  # B, N, P, d
            for layer in self.encoder:
                encoder_input = layer(encoder_input)
            hidden_states_unmasked = encoder_input

        return hidden_states_unmasked, unmasked_token_index, masked_token_index

    def decoding(self, hidden_states_unmasked, unmasked_token_index, masked_token_index):
        batch_size, num_nodes, num_time, _ = hidden_states_unmasked.shape  # B, N, P, d

        hidden_states_masked = self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), self.embedding_dim)
        hidden_states_unmasked = self.enc_2_dec(hidden_states_unmasked)

        if self.tod_embedding_dim > 0:
            hidden_states_masked = hidden_states_masked + self.tod[..., masked_token_index, :]
            hidden_states_unmasked = hidden_states_unmasked + self.tod[..., unmasked_token_index, :]

        if self.dow_embedding_dim > 0:
            hidden_states_masked = hidden_states_masked + self.dow[..., masked_token_index, :]
            hidden_states_unmasked = hidden_states_unmasked + self.dow[..., unmasked_token_index, :]
            
        if self.adaptive_embedding_dim > 0:
            hidden_states_masked = hidden_states_masked + self.adp[..., masked_token_index, :]
            hidden_states_unmasked = hidden_states_unmasked + self.adp[..., unmasked_token_index, :]
        

        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)
        hidden_states_full = self.decoder(hidden_states_full)
        reconstruction_full = self.output_layer(hidden_states_full)

        return reconstruction_full

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, masked_token_index):
        batch_size, num_nodes, num_time, _ = reconstruction_full.shape  # B, N, P, L
        reconstruction_masked_tokens = reconstruction_full[:, :, -len(masked_token_index) :, :]  # B, N, r*P, d
        label_full = real_value_full.unfold(1, self.patch_size, self.patch_size)[:, :, :, 0, :].transpose(1, 2)  # B, N, P, L
        label_masked_tokens = label_full[:, :, masked_token_index, :]  # B, N, r*P, d
        return reconstruction_masked_tokens, label_masked_tokens

    def forward(self, history_data: torch.Tensor):
        if self.mode == "pre-train":
            # encoding
            hidden_states_unmasked, unmasked_token_index, masked_token_index = self.encoding(history_data)
            # decoding
            reconstruction_full = self.decoding(hidden_states_unmasked, unmasked_token_index, masked_token_index)
            # for subsequent loss computing
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(
                reconstruction_full, history_data, masked_token_index
            )

            return reconstruction_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _ = self.encoding(history_data, mask=False)
            return hidden_states_full
