import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ml.config import *
from ml.dataset import TransliterationDataset
from ml.vocab import load_vocab
from ml.models.transformer import TransformerSeq2Seq
from ml.utils import collate_fn, create_padding_mask, save_checkpoint
from ml.logger import get_logger

# -----------------
# Initialize logger
# -----------------
logger = get_logger("train.py")
logger.info("Starting training script...")

# -----------------
# Load vocab and update config
# -----------------
src_char2idx, src_idx2char = load_vocab("src")
tgt_char2idx, tgt_idx2char = load_vocab("tgt")
SRC_VOCAB_SIZE = len(src_char2idx)
TGT_VOCAB_SIZE = len(tgt_char2idx)
logger.info(f"Loaded vocabularies - Source: {SRC_VOCAB_SIZE}, Target: {TGT_VOCAB_SIZE}")

# -----------------
# Prepare datasets and loaders
# -----------------
train_dataset = TransliterationDataset(PROCESSED_TRAIN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
logger.info(f"Training dataset loaded: {len(train_dataset)} examples, batch size: {BATCH_SIZE}")

test_dataset = TransliterationDataset(PROCESSED_TEST)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
logger.info(f"Test dataset loaded: {len(test_dataset)} examples, batch size: {BATCH_SIZE}")

# -----------------
# Initialize model
# -----------------
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
logger.info(f"Model initialized on device: {DEVICE}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
logger.info(f"Optimizer: Adam, Learning rate: {LR}")
logger.info(f"Loss function: CrossEntropyLoss with ignore_index={PAD_IDX}")

# -----------------
# Training loop
# -----------------
logger.info(f"Starting training for {NUM_EPOCHS} epochs...")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader, 1):
        src = batch["source"].to(DEVICE)
        tgt = batch["target"].to(DEVICE)

        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]

        src_mask = create_padding_mask(src)
        tgt_mask = create_padding_mask(tgt_input)

        optimizer.zero_grad()
        output = model(src, tgt_input, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
        output = output.view(-1, output.size(-1))
        tgt_labels = tgt_labels.contiguous().view(-1)

        loss = criterion(output, tgt_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 50 == 0:
            logger.info(f"Epoch {epoch} | Batch {i}/{len(train_loader)} | Current batch loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch [{epoch}/{NUM_EPOCHS}] completed. Average loss: {avg_loss:.4f}")

    # Save checkpoint every epoch
    checkpoint_path = f"transformer_epoch{epoch}.pth"
    save_checkpoint(model, optimizer, epoch, avg_loss, filename=checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

logger.info("Training completed successfully!")
