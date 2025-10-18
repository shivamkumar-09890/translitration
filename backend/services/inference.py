from ml.utils import load_model
from ml.vocab import load_vocab
import torch
from ml.config import *

# Load vocab and model once
src_char2idx, _ = load_vocab("src")
_, tgt_idx2char = load_vocab("tgt")
model = load_model("experiments/runs/transformer_epoch20.pth")

def greedy_decode(model, src_seq, max_len=MAX_LEN):
    src_seq = src_seq.unsqueeze(0).to(DEVICE)
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

    return ys.squeeze(0).tolist()

def transliterate_word(word: str) -> str:
    """
    Use model.predict(), but prepend SOS and append EOS to match training.
    """
    input_ids = [SOS_IDX] + [src_char2idx[ch] for ch in word] + [EOS_IDX]

    # Call wrapper's predict
    output_ids = model.predict(input_ids)

    # Convert to string
    output_word = "".join([tgt_idx2char[i] for i in output_ids if i != EOS_IDX and i != SOS_IDX])
    return output_word
