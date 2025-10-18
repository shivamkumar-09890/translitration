from ml.utils import load_model
from ml.vocab import src_char2idx, tgt_idx2char

# Load trained model once at startup
model = load_model("experiments/runs/best_model.pth")

def transliterate_word(word: str) -> str:
    # Convert input word to char indices
    input_ids = [src_char2idx[ch] for ch in word]
    # Run inference
    output_ids = model.predict(input_ids)
    # Convert back to string
    output_word = "".join([tgt_idx2char[idx] for idx in output_ids])
    return output_word
