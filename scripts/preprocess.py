import json
import os
import random
from ml.logger import get_logger

# ---------------------------
# Paths and constants
# ---------------------------
RAW_TRAIN = "data/raw/hin_train.json"
RAW_TEST = "data/raw/hin_test.json"
VOCAB_DIR = "data/vocab"
PROCESSED_DIR = "data/processed"

# Logger
logger = get_logger("DataPreprocessing")

# Special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

# Sample size
TRAIN_SAMPLE_SIZE = 100_000
HIGH_SCORE_RATIO = 0.6  # 60% top-score
NULL_SCORE_RATIO = 0.4  # 40% from null-score entries

# ---------------------------
# Helper functions
# ---------------------------
def load_json_lines(path):
    """Read JSON lines from file"""
    logger.info(f"Loading data from {path}")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} records from {path}")
    return data

def build_vocab(words, extra_tokens=[PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]):
    """Build character-level vocab from a list of words"""
    chars = set()
    for word in words:
        chars.update(list(word))
    vocab = extra_tokens + sorted(list(chars))
    char2idx = {ch: idx for idx, ch in enumerate(vocab)}
    idx2char = {idx: ch for ch, idx in char2idx.items()}
    logger.info(f"Built vocab of size {len(vocab)}")
    return vocab, char2idx, idx2char

def encode_words(words, char2idx):
    """Convert words to sequences of character indices with SOS/EOS tokens"""
    sequences = []
    for word in words:
        seq = [char2idx[SOS_TOKEN]] + [char2idx[ch] for ch in word] + [char2idx[EOS_TOKEN]]
        sequences.append(seq)
    return sequences

def save_vocab(char2idx, idx2char, prefix):
    os.makedirs(VOCAB_DIR, exist_ok=True)
    char2idx_path = os.path.join(VOCAB_DIR, f"{prefix}_char2idx.json")
    idx2char_path = os.path.join(VOCAB_DIR, f"{prefix}_idx2char.json")
    with open(char2idx_path, "w", encoding="utf-8") as f:
        json.dump(char2idx, f, ensure_ascii=False, indent=2)
    with open(idx2char_path, "w", encoding="utf-8") as f:
        json.dump(idx2char, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {prefix} vocab to {VOCAB_DIR}")

def save_processed_data(src_seq, tgt_seq, filename):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for s, t in zip(src_seq, tgt_seq):
            json.dump({"source": s, "target": t}, f)
            f.write("\n")
    logger.info(f"Saved processed data to {path} ({len(src_seq)} examples)")

# ---------------------------
# Main preprocessing
# ---------------------------
def main():
    logger.info("Starting data preprocessing...")

    # Load raw training data
    train_data = load_json_lines(RAW_TRAIN)

    # Separate entries with numeric score and null score
    data_with_score = [d for d in train_data if d["score"] is not None]
    data_null_score = [d for d in train_data if d["score"] is None]
    logger.info(f"{len(data_with_score)} entries with score, {len(data_null_score)} entries with null score")

    # Determine number of top-score and null-score samples
    n_top = int(TRAIN_SAMPLE_SIZE * HIGH_SCORE_RATIO)
    n_null = TRAIN_SAMPLE_SIZE - n_top
    logger.info(f"Selecting {n_top} top-score and {n_null} null-score examples")

    # Top-score examples
    top_data = sorted(data_with_score, key=lambda x: x["score"], reverse=True)[:n_top]
    logger.info(f"Top-score selection completed. Selected {len(top_data)} entries")

    # Random examples from null-score entries
    if len(data_null_score) < n_null:
        raise ValueError(f"Not enough null-score entries ({len(data_null_score)}) for {n_null} random selection")
    random_data = random.sample(data_null_score, n_null)
    logger.info(f"Random null-score selection completed. Selected {len(random_data)} entries")

    # Combine and shuffle
    train_data_final = top_data + random_data
    random.shuffle(train_data_final)
    logger.info("Combined top-score and null-score data and shuffled")

    # Source: Roman, Target: Devanagari
    src_train = [d["english word"] for d in train_data_final]
    tgt_train = [d["native word"] for d in train_data_final]

    # Load test data
    test_data = load_json_lines(RAW_TEST)
    src_test = [d["english word"] for d in test_data]
    tgt_test = [d["native word"] for d in test_data]
    logger.info(f"Test data loaded with {len(src_test)} examples")

    # Build vocab from train + test
    all_src = src_train + src_test
    all_tgt = tgt_train + tgt_test
    src_vocab, src_char2idx, src_idx2char = build_vocab(all_src)
    tgt_vocab, tgt_char2idx, tgt_idx2char = build_vocab(all_tgt)

    # Save vocab
    save_vocab(src_char2idx, src_idx2char, "src")
    save_vocab(tgt_char2idx, tgt_idx2char, "tgt")

    # Encode sequences
    src_train_seq = encode_words(src_train, src_char2idx)
    tgt_train_seq = encode_words(tgt_train, tgt_char2idx)
    src_test_seq = encode_words(src_test, src_char2idx)
    tgt_test_seq = encode_words(tgt_test, tgt_char2idx)
    logger.info("Encoding of sequences completed")

    # Save processed data
    save_processed_data(src_train_seq, tgt_train_seq, "train.json")
    save_processed_data(src_test_seq, tgt_test_seq, "test.json")

    logger.info("Data preprocessing completed successfully!")
    logger.info(f"Vocab size - Source (Roman): {len(src_vocab)}, Target (Devanagari): {len(tgt_vocab)}")
    logger.info(f"Processed files saved in {PROCESSED_DIR}, vocab saved in {VOCAB_DIR}")
    logger.info(f"Selected {len(src_train_seq)} train examples (60% top-score, 40% null-score random)")

if __name__ == "__main__":
    main()
