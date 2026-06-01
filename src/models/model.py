import math

import torch
from torch import nn


# Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html,
# only modified to account for "batch first".
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the given tensor.

        Args:
            x: tensor to add PE to [bs, seq_len, embed_dim]

        Returns:
            torch.Tensor: tensor with PE [bs, seq_len, embed_dim]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerWithPE(torch.nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, embed_dim: int, num_heads: int, num_layers: int
    ) -> None:
        """Initializes a transformer model with positional encoding.

        Args:
            in_dim: number of input features
            out_dim: number of features to predict
            embed_dim: embed features to this dimension
            num_heads: number of transformer heads
            num_layers: number of encoder and decoder layers
        """
        super().__init__()

        self.positional_encoding = PositionalEncoding(embed_dim)

        self.encoder_embedding = torch.nn.Linear(in_features=in_dim, out_features=embed_dim)
        self.decoder_embedding = torch.nn.Linear(in_features=out_dim, out_features=embed_dim)

        self.output_layer = torch.nn.Linear(in_features=embed_dim, out_features=out_dim)

        self.transformer = torch.nn.Transformer(
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            d_model=embed_dim,
            batch_first=True,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward function of the model.

        Args:
            src: input sequence to the encoder [bs, src_seq_len, num_features]
            tgt: input sequence to the decoder [bs, tgt_seq_len, num_features]

        Returns:
            torch.Tensor: predicted sequence [bs, tgt_seq_len, feat_dim]
        """
        # Embed encoder input and add positional encoding.
        # [bs, src_seq_len, embed_dim]
        src = self.encoder_embedding(src)
        src = self.positional_encoding(src)

        # Generate mask to avoid attention to future outputs.
        # [tgt_seq_len, tgt_seq_len]
        # Must live on the same device as the inputs, otherwise the transformer
        # raises a device-mismatch error when running on CUDA.
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(
            tgt.device
        )
        # Embed decoder input and add positional encoding.
        # [bs, tgt_seq_len, embed_dim]
        tgt = self.decoder_embedding(tgt)
        tgt = self.positional_encoding(tgt)

        # Get prediction from transformer and map to output dimension.
        # [bs, tgt_seq_len, embed_dim]
        pred = self.transformer(src, tgt, tgt_mask=tgt_mask)
        pred = self.output_layer(pred)

        return pred

    def infer(self, src: torch.Tensor, tgt_len: int) -> torch.Tensor:
        """Runs inference with the model, meaning: predicts future values
        for an unknown sequence.
        For this, iteratively generate the next output token while
        feeding the already generated ones as input sequence to the decoder.

        Args:
            src: input to the encoder [bs, src_seq_len, num_features]
            tgt_len: desired length of the output

        Returns:
            torch.Tensor: inferred sequence
        """
        # Force eval mode and disable grad so dropout does not perturb the
        # autoregressive rollout (the perturbation would compound across steps),
        # mirroring LSTM.infer. The previous mode is restored afterwards so the
        # method is safe to call directly without side effects.
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                output = torch.zeros((src.shape[0], tgt_len + 1, src.shape[2])).to(src.device)
                output[:, 0] = src[:, -1]
                for i in range(tgt_len):
                    output[:, i + 1] = self.forward(src, output)[:, i]
        finally:
            self.train(was_training)

        return output[:, 1:]


class LSTM(nn.Module):
    """Sequence-to-sequence LSTM forecaster.

    An encoder LSTM summarises the history (`src`) into its final hidden/cell
    state, which initialises a decoder LSTM that emits the requested number of
    *future* steps. Training uses teacher forcing and inference is
    autoregressive, but both predict the same forecast horizon aligned with the
    target - unlike a per-timestep head applied over the encoder window, which
    would compare in-sample outputs against the future target.

    Assumes ``input_dim == output_dim`` (the univariate series is both the input
    and the target), which holds for this project (NUM_FEATURES).
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, output_sequence_length=None, tgt=None):
        """Encode `src`, then decode the forecast horizon.

        Args:
            src: history fed to the encoder [bs, src_seq_len, input_dim]
            output_sequence_length: number of future steps to generate. Required
                for the autoregressive path (``tgt is None``); ignored when
                teacher forcing, where the horizon is taken from ``tgt``.
            tgt: optional teacher-forcing decoder input
                [bs, horizon, output_dim] - the target shifted right by one,
                starting at the last observed value. When given, the decoder is
                fed the ground truth instead of its own predictions.

        Returns:
            torch.Tensor: forecast [bs, horizon, output_dim]
        """
        _, (h, c) = self.encoder(src)

        if tgt is not None:
            # Teacher forcing: feed the shifted ground truth in one shot.
            dec_out, _ = self.decoder(tgt, (h, c))
            return self.fc(dec_out)

        if output_sequence_length is None:
            raise ValueError("output_sequence_length is required when tgt is None")

        # Autoregressive: seed with the last observed value, then feed each
        # prediction back in as the next decoder input.
        decoder_input = src[:, -1:, :]
        outputs = []
        for _ in range(output_sequence_length):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            step = self.fc(out)
            outputs.append(step)
            decoder_input = step
        return torch.cat(outputs, dim=1)

    def infer(self, src, sequence_length):
        # Self-contained like TransformerWithPE.infer: eval + no_grad so dropout
        # does not perturb the autoregressive rollout, restoring the prior mode.
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self.forward(src, output_sequence_length=sequence_length)
        finally:
            self.train(was_training)
