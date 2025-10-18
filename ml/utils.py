import random
import numpy as np
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from ml.config import PAD_IDX, CHECKPOINT_DIR
from ml.models.transformer import TransformerSeq2Seq
from ml.vocab import load_vocab
from ml.config import *
# -----------------
# Collate function for DataLoader
# -----------------
def collate_fn(batch):
    src_batch = [torch.tensor(item["source"], dtype=torch.long) for item in batch]
    tgt_batch = [torch.tensor(item["target"], dtype=torch.long) for item in batch]

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)

    return {"source": src_batch, "target": tgt_batch}

# -----------------
# Padding masks for Transformer
# -----------------
def create_padding_mask(seq):
    """
    seq: (batch_size, seq_len)
    Returns mask where True indicates padding
    """
    return seq == PAD_IDX  # (batch_size, seq_len)

# -----------------
# Save model checkpoint
# -----------------
def save_checkpoint(model, optimizer, epoch, loss, filename="model_checkpoint.pth"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)
    print(f"Checkpoint saved at {path}")

# Greedy decode function (for inference)
def greedy_decode(model, src_seq, max_len=MAX_LEN):
    src_seq = torch.tensor(src_seq, dtype=torch.long).unsqueeze(0).to(DEVICE)  # (1, seq_len)
    memory = model.encode(src_seq)

    ys = torch.tensor([[SOS_IDX]], device=DEVICE)

    for _ in range(max_len):
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask=tgt_mask)
        out = model.fc_out(out[:, -1])
        next_token = out.argmax(-1).unsqueeze(0)
        ys = torch.cat([ys, next_token], dim=1)
        if next_token.item() == EOS_IDX:
            break

    return ys.squeeze(0).tolist()[1:]  # remove SOS


def load_model(checkpoint_path: str):
    """
    Load the trained TransformerSeq2Seq model along with vocab.
    Returns a model wrapper with a `predict` method for inference.
    """

    # ------------------------
    # Load vocab
    # ------------------------
    src_char2idx, src_idx2char = load_vocab("src")
    tgt_char2idx, tgt_idx2char = load_vocab("tgt")

    SRC_VOCAB_SIZE = len(src_char2idx)
    TGT_VOCAB_SIZE = len(tgt_char2idx)

    # ------------------------
    # Initialize model
    # ------------------------
    model = TransformerSeq2Seq(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=MAX_LEN,
    ).to(DEVICE)

    # ------------------------
    # Load checkpoint
    # ------------------------
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ------------------------
    # Wrap with predict() method
    # ------------------------
    class ModelWrapper:
        def __init__(self, model, tgt_idx2char):
            self.model = model
            self.tgt_idx2char = tgt_idx2char

        def predict(self, input_ids):
            """
            input_ids: list of source char indices (no SOS/EOS added yet)
            """
            output_ids = greedy_decode(self.model, input_ids)
            return output_ids

    return ModelWrapper(model, tgt_idx2char)

def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False