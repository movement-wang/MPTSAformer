from torch import nn


class PatchEmbedding(nn.Module):

    def __init__(self, patch_size, in_channel, embed_dim):
        super().__init__()
        self.output_channel = embed_dim
        self.len_patch = patch_size  # the L
        self.output_channel = embed_dim
        self.input_embedding = nn.Conv1d(in_channel, embed_dim, kernel_size=self.len_patch, stride=self.len_patch)

    def forward(self, long_term_history):
        long_term_history = long_term_history.permute(0, 1, 3, 2)  # B, N, C, L
        batch_size, num_nodes, num_feat, len_time_series = long_term_history.shape
        long_term_history = long_term_history.reshape(batch_size * num_nodes, num_feat, len_time_series)  # B*N, C, L
        output = self.input_embedding(long_term_history)  # B*N, d, L/P
        output = output.view(batch_size, num_nodes, self.output_channel, -1)  # B, N, d, P
        assert output.shape[-1] == len_time_series / self.len_patch
        return output
