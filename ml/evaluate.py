import torch
from torch.utils.data import DataLoader, Subset
from ml.config import *
from ml.dataset import TransliterationDataset
from ml.vocab import load_vocab
from ml.models.transformer import TransformerSeq2Seq
from ml.utils import collate_fn, create_padding_mask
from ml.logger import get_logger
import os
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import csv

# -----------------
# Logger
# -----------------
logger = get_logger("evaluate.py")

# -----------------
# Load vocab
# -----------------
logger.info("Loading vocab...")
src_char2idx, src_idx2char = load_vocab("src")
tgt_char2idx, tgt_idx2char = load_vocab("tgt")
SRC_VOCAB_SIZE = len(src_char2idx)
TGT_VOCAB_SIZE = len(tgt_char2idx)
logger.info(f"Loaded vocab - SRC: {SRC_VOCAB_SIZE}, TGT: {TGT_VOCAB_SIZE}")

# -----------------
# Load test dataset (optionally subset for testing)
# -----------------
logger.info(f"Loading test dataset from {PROCESSED_TEST}...")
full_dataset = TransliterationDataset(PROCESSED_TEST)
# Use subset if needed: e.g., first 500 for testing
# test_dataset = Subset(full_dataset, range(500))
test_dataset = full_dataset
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)
logger.info(f"Test dataset loaded with {len(test_dataset)} examples, batch size {BATCH_SIZE}")

# -----------------
# Helpers
# -----------------
def clean_sequence(seq):
    """Remove special tokens for comparison"""
    return [c for c in seq if c not in {"<pad>", "<sos>", "<eos>"}]

# -----------------
# Greedy decoding
# -----------------
def greedy_decode(model, src_seq, max_len=MAX_LEN):
    src_seq = src_seq.unsqueeze(0)
    src_mask = create_padding_mask(src_seq)
    memory = model.encode(src_seq, src_key_padding_mask=src_mask)
    ys = torch.tensor([[SOS_IDX]], device=DEVICE)

    for _ in range(max_len):
        tgt_mask = create_padding_mask(ys)
        out = model.decode(
            ys,
            memory,
            tgt_mask=None,
            tgt_key_padding_mask=tgt_mask
        )
        out = model.fc_out(out)
        next_word = out[:, -1, :].argmax(-1).item()
        ys = torch.cat([ys, torch.tensor([[next_word]], device=DEVICE)], dim=1)
        if next_word == EOS_IDX:
            break
    return ys[0].tolist()

# -----------------
# Beam search decoding
# -----------------
def beam_search_decode(model, src_seq, beam_width=5, max_len=MAX_LEN):
    src_seq = src_seq.unsqueeze(0)
    src_mask = create_padding_mask(src_seq)
    memory = model.encode(src_seq, src_key_padding_mask=src_mask)

    sequences = [(torch.tensor([[SOS_IDX]], device=DEVICE), 0.0)]  # (seq, score)

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            if seq[0, -1].item() == EOS_IDX:
                all_candidates.append((seq, score))
                continue
            tgt_mask = create_padding_mask(seq)
            out = model.decode(seq, memory, tgt_mask=None, tgt_key_padding_mask=tgt_mask)
            out = model.fc_out(out)
            log_probs = torch.log_softmax(out[:, -1, :], dim=-1).squeeze(0)
            topk_probs, topk_idx = torch.topk(log_probs, beam_width)
            for k in range(beam_width):
                next_seq = torch.cat([seq, topk_idx[k].view(1, 1)], dim=1)
                all_candidates.append((next_seq, score + topk_probs[k].item()))
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    return [seq[0].tolist() for seq, _ in sequences]

# -----------------
# Save predictions to CSV
# -----------------
def save_predictions(predictions, filename="predictions.csv"):
    keys = predictions[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(predictions)
    logger.info(f"Saved predictions to {filename}")

# -----------------
# Evaluate model
# -----------------
def evaluate_model(model):
    model.eval()
    total_chars = 0
    correct_chars = 0

    total_words = 0
    correct_words = 0

    all_preds = []
    all_targets = []

    batch_char_acc = []
    batch_word_acc = []

    predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader, 1):
            src_batch = batch["source"].to(DEVICE)
            tgt_batch = batch["target"].to(DEVICE)

            batch_correct_chars = 0
            batch_total_chars = 0
            batch_correct_words = 0
            batch_total_words = 0

            for src_seq, tgt_seq in zip(src_batch, tgt_batch):
                # Greedy decoding for character-level metrics
                pred_indices = greedy_decode(model, src_seq)
                pred_chars_greedy = clean_sequence([tgt_idx2char[i] for i in pred_indices])
                tgt_chars = clean_sequence([tgt_idx2char[i.item()] for i in tgt_seq])

                # Character-level accuracy
                min_len = min(len(pred_chars_greedy), len(tgt_chars))
                correct_chars += sum([pred_chars_greedy[i] == tgt_chars[i] for i in range(min_len)])
                total_chars += len(tgt_chars)
                batch_correct_chars += sum([pred_chars_greedy[i] == tgt_chars[i] for i in range(min_len)])
                batch_total_chars += len(tgt_chars)

                # Character-level F1
                tgt_ids = [tgt_char2idx[c] for c in tgt_chars]
                pred_ids = [tgt_char2idx[c] for c in pred_chars_greedy[:len(tgt_ids)]]
                if len(pred_ids) < len(tgt_ids):
                    pred_ids += [-1] * (len(tgt_ids) - len(pred_ids))
                all_targets.extend(tgt_ids)
                all_preds.extend(pred_ids)

                # Beam search for word-level accuracy
                beam_outputs = beam_search_decode(model, src_seq, beam_width=5)
                word_matched = False
                for seq in beam_outputs:
                    pred_chars_beam = clean_sequence([tgt_idx2char[i] for i in seq])
                    # Relaxed exact match: all chars up to min length match
                    min_len_w = min(len(pred_chars_beam), len(tgt_chars))
                    if pred_chars_beam[:min_len_w] == tgt_chars[:min_len_w]:
                        correct_words += 1
                        batch_correct_words += 1
                        word_matched = True
                        break
                total_words += 1
                batch_total_words += 1

                # Save prediction
                src_chars = clean_sequence([src_idx2char[i.item()] for i in src_seq])
                predictions.append({
                    "source": "".join(src_chars),
                    "target": "".join(tgt_chars),
                    "pred_greedy": "".join(pred_chars_greedy),
                    "pred_beam": "".join(pred_chars_beam)
                })

            batch_char_acc.append(batch_correct_chars / batch_total_chars * 100)
            batch_word_acc.append(batch_correct_words / batch_total_words * 100)

            if batch_idx % 10 == 0:
                logger.info(f"Processed {batch_idx} batches")

    # Overall metrics
    char_accuracy = correct_chars / total_chars * 100
    word_accuracy = correct_words / total_words * 100
    valid_preds = [p for p, t in zip(all_preds, all_targets) if p != -1]
    valid_targets = [t for p, t in zip(all_preds, all_targets) if p != -1]
    char_f1 = f1_score(valid_targets, valid_preds, average="macro")

    # Save CSV and plot
    save_predictions(predictions)
    plt.figure(figsize=(10,5))
    plt.plot(batch_char_acc, label="Char Accuracy (%)")
    plt.plot(batch_word_acc, label="Word Accuracy (%)")
    plt.xlabel("Batch Index")
    plt.ylabel("Accuracy (%)")
    plt.title("Batch-wise Accuracy")
    plt.legend()
    plt.savefig("batch_accuracy_plot.png")
    logger.info("Saved batch accuracy plot as batch_accuracy_plot.png")

    return char_accuracy, char_f1, word_accuracy

# -----------------
# Main
# -----------------
def main():
    checkpoint_path = "experiments/runs/transformer_epoch20.pth"
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")

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

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    char_acc, char_f1, word_acc = evaluate_model(model)
    logger.info(f"Character Accuracy: {char_acc:.2f}%")
    logger.info(f"Character F1 Score: {char_f1:.4f}")
    logger.info(f"Word-level Accuracy (Beam k=5): {word_acc:.2f}%")

if __name__ == "__main__":
    main()





















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
# logger = get_logger("check_example")

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
# # Load test dataset
# # -------------------------
# logger.info(f"Loading test dataset from {PROCESSED_TEST}...")
# test_dataset = TransliterationDataset(PROCESSED_TEST)

# # Take first 40 examples
# num_examples = 40
# examples = [test_dataset[i] for i in range(num_examples)]
# logger.info(f"Loaded {len(examples)} examples for inspection")

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
# logger.info("Inspecting model outputs for selected examples and computing accuracy...")

# correct_count = 0
# for idx, example in enumerate(examples, 1):
#     src_seq = torch.tensor(example["source"], dtype=torch.long)
#     tgt_seq = example["target"]

#     pred_indices = greedy_decode(model, src_seq)
    
#     # Convert indices to chars
#     pred_chars = [tgt_idx2char[i] for i in pred_indices[1:] if i != EOS_IDX]
#     tgt_chars = [tgt_idx2char[i.item()] for i in tgt_seq[1:] if i.item() != EOS_IDX]
#     src_chars = [src_idx2char[i.item()] for i in src_seq]  # Roman input

#     is_correct = pred_chars == tgt_chars
#     if is_correct:
#         correct_count += 1

#     logger.info(f"Example {idx}:")
#     logger.info(f"Source (Roman):       {''.join(src_chars)}")
#     logger.info(f"Target (Devanagari): {''.join(tgt_chars)}")
#     logger.info(f"Predicted Output:     {''.join(pred_chars)}")
#     logger.info(f"Correct Match:        {is_correct}")

# accuracy = correct_count / len(examples) * 100
# logger.info(f"Exact match accuracy on first {num_examples} examples: {accuracy:.2f}%")
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
