import torch
import torch.nn as nn
import math
from ml.logger import get_logger

# -------------------------
# Logger
# -------------------------
logger = get_logger("model.Transformer")

# -------------------------
# Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)
        logger.info(f"PositionalEncoding initialized with d_model={d_model}, max_len={max_len}")

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# -------------------------
# Transformer Seq2Seq Model
# -------------------------
class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        max_len=200,
        device=None,
    ):
        super(TransformerSeq2Seq, self).__init__()
        self.d_model = d_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            f"Initializing TransformerSeq2Seq: src_vocab={src_vocab_size}, tgt_vocab={tgt_vocab_size}, "
            f"d_model={d_model}, nhead={nhead}, enc_layers={num_encoder_layers}, dec_layers={num_decoder_layers}, "
            f"dim_feedforward={dim_feedforward}, dropout={dropout}, max_len={max_len}, device={self.device}"
        )

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output layer
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.to(self.device)
        logger.info("TransformerSeq2Seq model initialized and moved to device successfully")

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        logger.debug(f"Forward called: src={src.shape}, tgt={tgt.shape}")
        src = src.to(self.device)
        tgt = tgt.to(self.device)

        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_decoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        logger.debug(f"Forward output shape: {output.shape}")
        return self.fc_out(output)

    def encode(self, src, src_key_padding_mask=None):
        logger.debug(f"Encoding src: {src.shape}")
        src = src.to(self.device)
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        logger.debug(f"Encoded memory shape: {memory.shape}")
        return memory

    def decode(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        logger.debug(f"Decoding tgt: {tgt.shape}, memory: {memory.shape}")
        tgt = tgt.to(self.device)
        tgt_emb = self.pos_decoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        out = self.transformer.decoder(
            tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )
        logger.debug(f"Decoder output shape: {out.shape}")
        return out

    def predict(self, src, max_len=30, sos_token=1, eos_token=2):
        logger.info(f"Greedy decoding started: max_len={max_len}, sos_token={sos_token}, eos_token={eos_token}")
        self.eval()
        src = src.to(self.device)
        with torch.no_grad():
            memory = self.encode(src)
            tgt = torch.tensor([[sos_token]], device=self.device)

            for step in range(max_len):
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.device)
                out = self.decode(tgt, memory, tgt_mask=tgt_mask)
                out = self.fc_out(out[:, -1])
                next_token = out.argmax(-1).unsqueeze(0)
                tgt = torch.cat([tgt, next_token], dim=1)

                logger.debug(f"Step {step}: next_token={next_token.item()}")

                if next_token.item() == eos_token:
                    logger.info(f"EOS token reached at step {step}")
                    break

        decoded_sequence = tgt.squeeze(0).tolist()
        logger.info(f"Greedy decoding completed: output_len={len(decoded_sequence)}, device={self.device}")
        return decoded_sequence
