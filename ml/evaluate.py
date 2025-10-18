# import torch
# from torch.utils.data import DataLoader
# from ml.config import *
# from ml.dataset import TransliterationDataset
# from ml.vocab import load_vocab
# from ml.models.transformer import TransformerSeq2Seq
# from ml.utils import collate_fn, create_padding_mask
# from ml.logger import get_logger
# import glob
# import os

# # -----------------
# # Logger
# # -----------------
# logger = get_logger("evaluate.py")

# # -----------------
# # Load vocab
# # -----------------
# logger.info("Loading vocab...")
# src_char2idx, src_idx2char = load_vocab("src")
# tgt_char2idx, tgt_idx2char = load_vocab("tgt")
# SRC_VOCAB_SIZE = len(src_char2idx)
# TGT_VOCAB_SIZE = len(tgt_char2idx)
# logger.info(f"Loaded vocab - SRC: {SRC_VOCAB_SIZE}, TGT: {TGT_VOCAB_SIZE}")

# # -----------------
# # Load test dataset
# # -----------------
# logger.info(f"Loading test dataset from {PROCESSED_TEST}...")
# test_dataset = TransliterationDataset(PROCESSED_TEST)
# test_loader = DataLoader(
#     test_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     collate_fn=collate_fn
# )
# logger.info(f"Test dataset loaded with {len(test_dataset)} examples, batch size {BATCH_SIZE}")

# # -----------------
# # Greedy decoding
# # -----------------
# def greedy_decode(model, src_seq, max_len=MAX_LEN):
#     """Greedy decode single sequence"""
#     src_seq = src_seq.unsqueeze(0)  # (1, seq_len)
#     src_mask = create_padding_mask(src_seq)
#     memory = model.encode(src_seq, src_key_padding_mask=src_mask)
#     ys = torch.tensor([[SOS_IDX]], device=DEVICE)

#     for _ in range(max_len):
#         tgt_mask = create_padding_mask(ys)
#         out = model.decode(
#             ys,
#             memory,
#             tgt_mask=None,
#             tgt_key_padding_mask=tgt_mask
#         )
#         out = model.fc_out(out)
#         next_word = out[:, -1, :].argmax(-1).item()
#         ys = torch.cat([ys, torch.tensor([[next_word]], device=DEVICE)], dim=1)
#         if next_word == EOS_IDX:
#             break
#     return ys[0].tolist()

# # -----------------
# # Evaluate a single model
# # -----------------
# def evaluate_model(model):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(test_loader, 1):
#             # Fix tensor warnings
#             src_batch = batch["source"].to(DEVICE)
#             tgt_batch = batch["target"].to(DEVICE)

#             for src_seq, tgt_seq in zip(src_batch, tgt_batch):
#                 pred_indices = greedy_decode(model, src_seq)
#                 pred_chars = [tgt_idx2char[i] for i in pred_indices[1:] if i != EOS_IDX]
#                 tgt_chars = [tgt_idx2char[i.item()] for i in tgt_seq[1:] if i.item() != EOS_IDX]
#                 if pred_chars == tgt_chars:
#                     correct += 1
#                 total += 1

#             if batch_idx % 10 == 0:
#                 logger.info(f"Processed {batch_idx} batches, current accuracy: {correct / total * 100:.2f}%")

#     accuracy = correct / total * 100
#     return accuracy

# # -----------------
# # Main: Evaluate multiple checkpoints and select best
# # -----------------
# def main():
#     checkpoint_files = sorted(glob.glob("experiments/runs/transformer_epoch*.pth"))
#     logger.info(f"Found {len(checkpoint_files)} checkpoints: {checkpoint_files}")

#     best_accuracy = -1
#     best_model_path = None

#     for ckpt_path in checkpoint_files:
#         logger.info(f"Evaluating checkpoint: {ckpt_path}")
#         model = TransformerSeq2Seq(
#             src_vocab_size=SRC_VOCAB_SIZE,
#             tgt_vocab_size=TGT_VOCAB_SIZE,
#             d_model=D_MODEL,
#             nhead=NHEAD,
#             num_encoder_layers=NUM_ENCODER_LAYERS,
#             num_decoder_layers=NUM_DECODER_LAYERS,
#             dim_feedforward=DIM_FEEDFORWARD,
#             dropout=DROPOUT,
#             max_len=MAX_LEN,
#         ).to(DEVICE)

#         checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
#         model.load_state_dict(checkpoint["model_state_dict"])
#         model.eval()

#         accuracy = evaluate_model(model)
#         logger.info(f"Checkpoint {ckpt_path} accuracy: {accuracy:.2f}%")

#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_model_path = ckpt_path

#     if best_model_path:
#         logger.info(f"Best model: {best_model_path} with accuracy {best_accuracy:.2f}%")
#         best_model = TransformerSeq2Seq(
#             src_vocab_size=SRC_VOCAB_SIZE,
#             tgt_vocab_size=TGT_VOCAB_SIZE,
#             d_model=D_MODEL,
#             nhead=NHEAD,
#             num_encoder_layers=NUM_ENCODER_LAYERS,
#             num_decoder_layers=NUM_DECODER_LAYERS,
#             dim_feedforward=DIM_FEEDFORWARD,
#             dropout=DROPOUT,
#             max_len=MAX_LEN,
#         ).to(DEVICE)
#         checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=True)
#         best_model.load_state_dict(checkpoint["model_state_dict"])
#         os.makedirs("experiments/runs", exist_ok=True)
#         torch.save({"model_state_dict": best_model.state_dict()}, "experiments/runs/best_model.pth")
#         logger.info("Saved best model as experiments/runs/best_model.pth")

# if __name__ == "__main__":
#     main()


import torch
from torch.utils.data import DataLoader
from ml.dataset import TransliterationDataset
from ml.vocab import load_vocab
from ml.models.transformer import TransformerSeq2Seq
from ml.utils import collate_fn
from ml.logger import get_logger
from ml.config import *

# -------------------------
# Logger
# -------------------------
logger = get_logger("check_example")

# -------------------------
# Load vocab
# -------------------------
logger.info("Loading vocab...")
src_char2idx, src_idx2char = load_vocab("src")
tgt_char2idx, tgt_idx2char = load_vocab("tgt")

SRC_VOCAB_SIZE = len(src_char2idx)
TGT_VOCAB_SIZE = len(tgt_char2idx)
logger.info(f"Loaded vocab - SRC: {SRC_VOCAB_SIZE}, TGT: {TGT_VOCAB_SIZE}")

# -------------------------
# Load test dataset
# -------------------------
logger.info(f"Loading test dataset from {PROCESSED_TEST}...")
test_dataset = TransliterationDataset(PROCESSED_TEST)

# Take first 40 examples
num_examples = 40
examples = [test_dataset[i] for i in range(num_examples)]
logger.info(f"Loaded {len(examples)} examples for inspection")

# -------------------------
# Load trained model (epoch 20)
# -------------------------
checkpoint_path = "experiments/runs/transformer_epoch20.pth"
logger.info(f"Loading trained model from {checkpoint_path}...")
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

checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
logger.info("Model loaded and set to evaluation mode")

# -------------------------
# Greedy decode function
# -------------------------
def greedy_decode(model, src_seq, max_len=MAX_LEN):
    src_seq = src_seq.unsqueeze(0).to(DEVICE)  # (1, seq_len)
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

# -------------------------
# Inspect outputs and compute accuracy
# -------------------------
logger.info("Inspecting model outputs for selected examples and computing accuracy...")

correct_count = 0
for idx, example in enumerate(examples, 1):
    src_seq = torch.tensor(example["source"], dtype=torch.long)
    tgt_seq = example["target"]

    pred_indices = greedy_decode(model, src_seq)
    
    # Convert indices to chars
    pred_chars = [tgt_idx2char[i] for i in pred_indices[1:] if i != EOS_IDX]
    tgt_chars = [tgt_idx2char[i.item()] for i in tgt_seq[1:] if i.item() != EOS_IDX]
    src_chars = [src_idx2char[i.item()] for i in src_seq]  # Roman input

    is_correct = pred_chars == tgt_chars
    if is_correct:
        correct_count += 1

    logger.info(f"Example {idx}:")
    logger.info(f"Source (Roman):       {''.join(src_chars)}")
    logger.info(f"Target (Devanagari): {''.join(tgt_chars)}")
    logger.info(f"Predicted Output:     {''.join(pred_chars)}")
    logger.info(f"Correct Match:        {is_correct}")

accuracy = correct_count / len(examples) * 100
logger.info(f"Exact match accuracy on first {num_examples} examples: {accuracy:.2f}%")
# import torch
# from torch.utils.data import DataLoader
# from ml.dataset import TransliterationDataset
# from ml.vocab import load_vocab
# from ml.models.transformer import TransformerSeq2Seq
# from ml.utils import collate_fn
# from ml.logger import get_logger
# from ml.config import *

# # -------------------------
# # Logger
# # -------------------------
# logger = get_logger("check_train_example")

# # -------------------------
# # Load vocab
# # -------------------------
# logger.info("Loading vocab...")
# src_char2idx, src_idx2char = load_vocab("src")
# tgt_char2idx, tgt_idx2char = load_vocab("tgt")

# SRC_VOCAB_SIZE = len(src_char2idx)
# TGT_VOCAB_SIZE = len(tgt_char2idx)
# logger.info(f"Loaded vocab - SRC: {SRC_VOCAB_SIZE}, TGT: {TGT_VOCAB_SIZE}")

# # -------------------------
# # Load train dataset
# # -------------------------
# logger.info(f"Loading train dataset from {PROCESSED_TRAIN}...")
# train_dataset = TransliterationDataset(PROCESSED_TRAIN)

# # Take first 20 examples (can change to 40 if you want)
# num_examples = 40
# examples = [train_dataset[i] for i in range(num_examples)]
# logger.info(f"Loaded {len(examples)} examples for inspection from train set")

# # -------------------------
# # Load trained model (epoch 20)
# # -------------------------
# checkpoint_path = "experiments/runs/transformer_epoch20.pth"
# logger.info(f"Loading trained model from {checkpoint_path}...")
# model = TransformerSeq2Seq(
#     src_vocab_size=SRC_VOCAB_SIZE,
#     tgt_vocab_size=TGT_VOCAB_SIZE,
#     d_model=D_MODEL,
#     nhead=NHEAD,
#     num_encoder_layers=NUM_ENCODER_LAYERS,
#     num_decoder_layers=NUM_DECODER_LAYERS,
#     dim_feedforward=DIM_FEEDFORWARD,
#     dropout=DROPOUT,
#     max_len=MAX_LEN,
# ).to(DEVICE)

# checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval()
# logger.info("Model loaded and set to evaluation mode")

# # -------------------------
# # Greedy decode function
# # -------------------------
# def greedy_decode(model, src_seq, max_len=MAX_LEN):
#     src_seq = src_seq.unsqueeze(0).to(DEVICE)  # (1, seq_len)
#     memory = model.encode(src_seq)

#     ys = torch.tensor([[SOS_IDX]], device=DEVICE)

#     for _ in range(max_len):
#         tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(DEVICE)
#         out = model.decode(ys, memory, tgt_mask=tgt_mask)
#         out = model.fc_out(out[:, -1])
#         next_token = out.argmax(-1).unsqueeze(0)
#         ys = torch.cat([ys, next_token], dim=1)
#         if next_token.item() == EOS_IDX:
#             break

#     return ys.squeeze(0).tolist()

# # -------------------------
# # Inspect outputs and compute accuracy
# # -------------------------
# logger.info("Inspecting model outputs for selected train examples and computing accuracy...")

# correct_count = 0
# for idx, example in enumerate(examples, 1):
#     src_seq = torch.tensor(example["source"], dtype=torch.long)
#     tgt_seq = example["target"]

#     pred_indices = greedy_decode(model, src_seq)
    
#     # Convert indices to chars
#     pred_chars = [tgt_idx2char[i] for i in pred_indices[1:] if i != EOS_IDX]
#     tgt_chars = [tgt_idx2char[i.item()] for i in tgt_seq[1:] if i.item() != EOS_IDX]
#     src_chars = [src_idx2char[i.item()] for i in src_seq]

#     is_correct = pred_chars == tgt_chars
#     if is_correct:
#         correct_count += 1

#     logger.info(f"Example {idx}:")
#     logger.info(f"Source (Roman):       {''.join(src_chars)}")
#     logger.info(f"Target (Devanagari): {''.join(tgt_chars)}")
#     logger.info(f"Predicted Output:     {''.join(pred_chars)}")
#     logger.info(f"Correct Match:        {is_correct}")

# accuracy = correct_count / len(examples) * 100
# logger.info(f"Exact match accuracy on first {num_examples} train examples: {accuracy:.2f}%")
