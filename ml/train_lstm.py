import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from ml.dataset import TransliterationDataset, collate_fn
from ml.models.lstm import Encoder, Decoder, Seq2Seq
from ml.utils import save_checkpoint, set_seed

PAD_IDX = 0  # Make sure this matches your preprocessing padding index


def train_one_epoch(model, dataloader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0

    for batch in dataloader:
        src = batch["source"].to(device)
        tgt = batch["target"].to(device)
        src_lengths = (src != PAD_IDX).sum(dim=1)

        optimizer.zero_grad()
        output = model(src, src_lengths, tgt, teacher_forcing_ratio=0.5)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)  # skip <sos>
        tgt_flat = tgt[:, 1:].reshape(-1)

        loss = criterion(output, tgt_flat)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch["source"].to(device)
            tgt = batch["target"].to(device)
            src_lengths = (src != PAD_IDX).sum(dim=1)

            output = model(src, src_lengths, tgt, teacher_forcing_ratio=0)  # no teacher forcing

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt_flat = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt_flat)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def main():
    set_seed(42)

    # -------- Config -------- #
    DATA_DIR = "data/processed"
    VOCAB_DIR = "data/vocab"
    SAVE_DIR = "experiments/runs"
    os.makedirs(SAVE_DIR, exist_ok=True)

    BATCH_SIZE = 128
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    DROPOUT = 0.3
    N_EPOCHS = 10
    CLIP = 1.0
    LR = 2e-3
    VAL_SPLIT = 0.1  # 10% validation split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Load vocab -------- #
    with open(os.path.join(VOCAB_DIR, "src_char2idx.json")) as f:
        src_char2idx = json.load(f)
    with open(os.path.join(VOCAB_DIR, "tgt_char2idx.json")) as f:
        tgt_char2idx = json.load(f)

    INPUT_DIM = len(src_char2idx)
    OUTPUT_DIM = len(tgt_char2idx)

    # -------- Load dataset (only train.json exists) -------- #
    full_data = TransliterationDataset(os.path.join(DATA_DIR, "train.json"))

    val_size = int(VAL_SPLIT * len(full_data))
    train_size = len(full_data) - val_size
    train_data, val_data = random_split(full_data, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn)

    # -------- Build model -------- #
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)  # ignore <pad> token

    best_val_loss = float("inf")

    # -------- Training loop -------- #
    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CLIP, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"[Epoch {epoch}/{N_EPOCHS}] "
              f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, filename="lstm_best.pth")

    print("Training complete.")


if __name__ == "__main__":
    main()
