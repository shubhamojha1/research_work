import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
from quaesita.PositionalEncoding import PositionalEncoding

class MultiHeadedCriticTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, input_sequence_length=1, forecasting_step=1, num_critics=4):
        super(MultiHeadedCriticTransformer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.input_sequence_length = input_sequence_length
        self.forecasting_step = forecasting_step
        self.num_critics = num_critics

        self.positional_encoding = PositionalEncoding(self.d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.encoder_norm = LayerNorm(self.d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=self.num_encoder_layers, norm=self.encoder_norm)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.decoder_norm = LayerNorm(self.d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=self.num_decoder_layers, norm=self.decoder_norm)

        self.critics = nn.ModuleList([nn.Linear(self.d_model, 1) for _ in range(self.num_critics)])

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for critic in self.critics:
            critic.bias.data.zero_()
            critic.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, tgt_mask):
        memory = self.encoder(self.positional_encoding(src))
        output = self.decoder(self.positional_encoding(tgt), memory, tgt_mask)

        critic_outputs = [critic(output).squeeze(-1) for critic in self.critics]
        combined_output = sum(critic_output for critic_output in critic_outputs)

        return combined_output

# Create an instance of the new model
model = MultiHeadedCriticTransformer()
