import torch
from ml.logger import get_logger
from ml.vocab import load_vocab
from ml.models.lstm import Encoder, Decoder, Seq2Seq

# -----------------
# Setup logger
# -----------------
logger = get_logger("inference_lstm")

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
SRC_VOCAB_SIZE = len(src_char2idx)
TGT_VOCAB_SIZE = len(tgt_char2idx)
logger.info(f"Loaded vocabularies - Source: {SRC_VOCAB_SIZE}, Target: {TGT_VOCAB_SIZE}")

# -----------------
# Model hyperparameters
# -----------------
EMB_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3

# -----------------
# Initialize model
# -----------------
encoder = Encoder(SRC_VOCAB_SIZE, EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = Decoder(TGT_VOCAB_SIZE, EMB_DIM, HIDDEN_DIM, DROPOUT)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

# -----------------
# Load checkpoint
# -----------------
checkpoint_path = "experiments/runs/lstm_best.pth"
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
logger.info(f"Model loaded from checkpoint: {checkpoint_path}")

# -----------------
# Input preprocessing
# -----------------
def preprocess_input(src_seq, src_char2idx):
    # Keep only allowed characters
    src_seq = [ch.lower() for ch in src_seq if ch.lower() in src_char2idx]
    src_idx = [src_char2idx["<sos>"]] + [src_char2idx[ch] for ch in src_seq] + [src_char2idx["<eos>"]]
    return src_idx

# -----------------
# Inference function
# -----------------
def predict(model, src_seq, src_char2idx, tgt_idx2char, max_len=12):
    logger.info(f"Predicting for input: {src_seq}")
    model.eval()
    
    src_idx = preprocess_input(src_seq, src_char2idx)
    src_tensor = torch.LongTensor([src_idx]).to(DEVICE)
    src_len = torch.LongTensor([len(src_idx)]).to(DEVICE)
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_len)
        mask = model.create_mask(src_tensor)
        
        input_token = torch.LongTensor([tgt_char2idx["<sos>"]]).to(DEVICE)
        output_seq = []

        last_token = None
        repeat_count = 0

        for step in range(max_len):
            output, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs, mask)
            top1 = output.argmax(1).item()
            
            if top1 == tgt_char2idx["<eos>"]:
                logger.info(f"Encountered <eos> at step {step}. Stopping decoding.")
                break

            # Stop if same token repeats more than 2 times consecutively
            if top1 == last_token:
                repeat_count += 1
                if repeat_count >= 2:
                    logger.warning(f"Token {tgt_idx2char[top1]} repeated {repeat_count} times. Stopping decoding.")
                    break
            else:
                repeat_count = 0

            output_seq.append(tgt_idx2char[top1])
            last_token = top1
            input_token = torch.LongTensor([top1]).to(DEVICE)

    predicted_text = "".join(output_seq)
    logger.info(f"Predicted output: {predicted_text}")
    return predicted_text

