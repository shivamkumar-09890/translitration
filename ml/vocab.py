import json
import os
from ml.config import VOCAB_DIR

def load_vocab(prefix="src"):
    """
    Loads char2idx and idx2char JSON files
    """
    char2idx_path = os.path.join(VOCAB_DIR, f"{prefix}_char2idx.json")
    idx2char_path = os.path.join(VOCAB_DIR, f"{prefix}_idx2char.json")

    with open(char2idx_path, "r", encoding="utf-8") as f:
        char2idx = json.load(f)
    with open(idx2char_path, "r", encoding="utf-8") as f:
        idx2char = json.load(f)

    # Convert idx keys back to int
    idx2char = {int(k): v for k, v in idx2char.items()}
    return char2idx, idx2char
