from torch import nn
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer


class PositionalEncoding(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_data, index=None, abs_idx=None):
        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        tp_enc_2d = PositionalEncoding2D(num_feat).to(input_data.device)
        input_data += tp_enc_2d(input_data)
        return input_data, tp_enc_2d(input_data)
