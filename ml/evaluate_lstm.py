import json
import torch
import logging
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from ml.vocab import load_vocab
from ml.models.lstm import Encoder, Decoder, Seq2Seq

# -----------------
# Logging
# -----------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [eval_lstm] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("eval_lstm")

# -----------------
# Device
# -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# -----------------
# Load vocab
# -----------------
src_char2idx, src_idx2char = load_vocab("src")
tgt_char2idx, tgt_idx2char = load_vocab("tgt")
logger.info(f"Source vocab size: {len(src_char2idx)}, Target vocab size: {len(tgt_char2idx)}")

# -----------------
# Initialize model
# -----------------
EMB_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3

encoder = Encoder(len(src_char2idx), EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = Decoder(len(tgt_char2idx), EMB_DIM, HIDDEN_DIM, DROPOUT)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

checkpoint = torch.load("experiments/runs/lstm_best.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
logger.info("Model loaded successfully!")

# -----------------
# Dataset class
# -----------------
class TransliterationEvalDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# -----------------
# Custom collate function for padding
# -----------------
def collate_fn(batch):
    src_batch = [torch.tensor(item["source"], dtype=torch.long) for item in batch]
    tgt_batch = [torch.tensor(item["target"], dtype=torch.long) for item in batch]

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=src_char2idx["<pad>"])
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_char2idx["<pad>"])

    return {"source": src_padded, "target": tgt_padded}

# -----------------
# Prediction function
# -----------------
def predict(src_idx, max_len=50):
    model.eval()
    src_tensor = torch.LongTensor([src_idx]).to(DEVICE)
    src_len = torch.LongTensor([len(src_idx)]).to(DEVICE)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_len)
        mask = model.create_mask(src_tensor)

        input_token = torch.LongTensor([tgt_char2idx["<sos>"]]).to(DEVICE)
        output_seq = []

        last_token = None
        repeat_count = 0

        for _ in range(max_len):
            output, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs, mask)
            top1 = output.argmax(1).item()

            if top1 == tgt_char2idx["<eos>"]:
                break

            # Avoid repeated tokens
            if top1 == last_token:
                repeat_count += 1
                if repeat_count >= 2:
                    break
            else:
                repeat_count = 0

            output_seq.append(top1)
            last_token = top1
            input_token = torch.LongTensor([top1]).to(DEVICE)

    return output_seq

# -----------------
# Load evaluation dataset
# -----------------
TEST_FILE = "data/processed/test.json"  # Replace with your test file path
dataset = TransliterationEvalDataset(TEST_FILE)
data_loader = DataLoader(dataset, batch_size=50, shuffle=False, collate_fn=collate_fn)
logger.info(f"Loaded {len(dataset)} examples for evaluation. Batch size: 50")

# -----------------
# Run evaluation
# -----------------
y_true = []
y_pred = []
results = []

for batch_idx, batch in enumerate(data_loader, 1):
    src_batch = batch["source"]
    tgt_batch = batch["target"]

    for i in range(src_batch.size(0)):
        src_idx = src_batch[i].tolist()
        tgt_idx = tgt_batch[i].tolist()[1:-1]  # remove <sos> and <eos>

        # Convert source indices to characters for prediction
        src_input_str = "".join([src_idx2char[i] for i in src_idx[1:-1]])  # skip <sos>/<eos>

        # Predict
        pred_idx = predict([src_char2idx.get(c, 0) for c in src_input_str], max_len=50)

        # Pad or truncate prediction to match target length
        if len(pred_idx) < len(tgt_idx):
            pred_idx += [tgt_char2idx["<pad>"]] * (len(tgt_idx) - len(pred_idx))
        else:
            pred_idx = pred_idx[:len(tgt_idx)]

        y_true.extend(tgt_idx)
        y_pred.extend(pred_idx)

        results.append({
            "source": src_input_str,
            "target": "".join([tgt_idx2char[i] for i in tgt_idx]),
            "prediction": "".join([tgt_idx2char[i] for i in pred_idx])
        })

    logger.info(f"Processed {(batch_idx) * data_loader.batch_size}/{len(dataset)} examples...")

# -----------------
# Compute char-level F1
# -----------------
f1 = f1_score(y_true, y_pred, average="micro")
logger.info(f"Character-level F1 score: {f1:.4f}")

# -----------------
# Save predictions
# -----------------
RESULT_FILE = "predictions.json"
with open(RESULT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
logger.info(f"Saved predictions to {RESULT_FILE}")
