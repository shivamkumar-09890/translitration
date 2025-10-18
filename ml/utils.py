import torch
import os

from torch.nn.utils.rnn import pad_sequence
from ml.config import PAD_IDX, CHECKPOINT_DIR

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
