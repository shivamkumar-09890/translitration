# # import torch
# # import torch.nn as nn
# # from torch.utils.data import Dataset, DataLoader
# # from torch.nn.utils.rnn import pad_sequence
# # import json

# # # ====== CONFIG ======
# # BATCH_SIZE = 64
# # EPOCHS = 5
# # HIDDEN_SIZE = 256
# # EMBEDDING_SIZE = 128
# # LEARNING_RATE = 0.001
# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # ====== FILES ======
# # train_file = "train_numeric.json"
# # valid_file = "valid_numeric.json"
# # test_file = "test_numeric.json"

# # # ====== DATASET ======
# # class NumericDataset(Dataset):
# #     def __init__(self, path):
# #         self.data = []
# #         with open(path, 'r', encoding='utf-8') as f:
# #             for line in f:
# #                 item = json.loads(line)
# #                 self.data.append(item)

# #     def __len__(self):
# #         return len(self.data)

# #     def __getitem__(self, idx):
# #         src = torch.tensor(self.data[idx]['src'], dtype=torch.long)
# #         tgt = torch.tensor(self.data[idx]['tgt'], dtype=torch.long)
# #         return src, tgt

# # # ====== COLLATE FUNCTION ======
# # def collate_fn(batch):
# #     src_batch, tgt_batch = zip(*batch)
# #     src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
# #     tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
# #     return src_batch.to(DEVICE), tgt_batch.to(DEVICE)

# # # ====== LOAD DATA ======
# # train_loader = DataLoader(NumericDataset(train_file), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
# # valid_loader = DataLoader(NumericDataset(valid_file), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# # # ====== MODEL ======
# # class LSTMSeq2Seq(nn.Module):
# #     def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, hidden_size):
# #         super().__init__()
# #         self.embedding_src = nn.Embedding(src_vocab_size, embed_size, padding_idx=0)
# #         self.embedding_tgt = nn.Embedding(tgt_vocab_size, embed_size, padding_idx=0)
# #         self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
# #         self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
# #         self.fc = nn.Linear(hidden_size, tgt_vocab_size)

# #     def forward(self, src, tgt):
# #         # Encoder
# #         embedded_src = self.embedding_src(src)
# #         _, (hidden, cell) = self.encoder(embedded_src)

# #         # Decoder
# #         embedded_tgt = self.embedding_tgt(tgt)
# #         output, _ = self.decoder(embedded_tgt, (hidden, cell))
# #         output = self.fc(output)
# #         return output

# # # ====== GET VOCAB SIZES ======
# # # Find max integer in numeric files to determine vocab size
# # def get_vocab_size(file_path):
# #     max_idx = 0
# #     with open(file_path, 'r', encoding='utf-8') as f:
# #         for line in f:
# #             item = json.loads(line)
# #             max_idx = max(max_idx, max(item['src']), max(item['tgt']))
# #     return max_idx + 1  # +1 for padding (0)

# # SRC_VOCAB_SIZE = get_vocab_size(train_file)
# # TGT_VOCAB_SIZE = get_vocab_size(train_file)

# # # ====== INIT MODEL ======
# # model = LSTMSeq2Seq(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE).to(DEVICE)
# # criterion = nn.CrossEntropyLoss(ignore_index=0)
# # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# # # ====== TRAINING LOOP ======
# # for epoch in range(EPOCHS):
# #     model.train()
# #     total_loss = 0
# #     for src_batch, tgt_batch in train_loader:
# #         optimizer.zero_grad()
# #         # Decoder input excludes the last token (teacher forcing)
# #         output = model(src_batch, tgt_batch[:, :-1])
# #         # Shift tgt_batch by 1 for target labels
# #         loss = criterion(output.reshape(-1, TGT_VOCAB_SIZE), tgt_batch[:, 1:].reshape(-1))
# #         loss.backward()
# #         optimizer.step()
# #         total_loss += loss.item()
# #     avg_loss = total_loss / len(train_loader)
# #     print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# # print("Training complete!")

# # File: ml/models/lstm_gpu.py

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torch.optim as optim
# import time

# # Check for GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # ---------------------- Dataset ----------------------
# class NumericDataset(Dataset):
#     def __init__(self, data_file):
#         # Load your preprocessed numeric data here
#         self.data = torch.load(data_file)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         x, y = self.data[idx]
#         return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# def collate_fn(batch):
#     xs, ys = zip(*batch)
#     return torch.stack(xs), torch.stack(ys)

# # ---------------------- Model ----------------------
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=2):
#         super(LSTMModel, self).__init__()
#         self.embedding = nn.Embedding(input_size, hidden_size)
#         self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         embedded = self.embedding(x)
#         output, _ = self.lstm(embedded)
#         logits = self.fc(output[:, -1, :])
#         return logits

# # ---------------------- Training ----------------------
# def train_model(train_file, input_size, hidden_size, output_size, epochs=10, batch_size=128, lr=1e-3):
#     dataset = NumericDataset(train_file)
#     train_loader = DataLoader(
#         dataset, batch_size=batch_size, shuffle=True, 
#         collate_fn=collate_fn, pin_memory=True
#     )

#     model = LSTMModel(input_size, hidden_size, output_size).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     print("Starting training...\n")

#     for epoch in range(epochs):
#         start_time = time.time()
#         total_loss = 0
#         for x_batch, y_batch in train_loader:
#             x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)

#             optimizer.zero_grad()
#             outputs = model(x_batch)
#             loss = criterion(outputs, y_batch)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         epoch_time = time.time() - start_time
#         print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f} | Time: {epoch_time:.2f}s")

#     torch.save(model.state_dict(), "ml/models/lstm_trained.pth")
#     print("Training complete. Model saved at ml/models/lstm_trained.pth")

# # ---------------------- Entry Point ----------------------
# if __name__ == "__main__":
#     train_file = "ml/data/train_data.pt"  # change to your file path
#     train_model(
#         train_file=train_file,
#         input_size=1000,   # adjust as per vocab size
#         hidden_size=256,
#         output_size=1000,  # adjust as per output vocab size
#         epochs=10,
#         batch_size=128,
#         lr=0.001
#     )






# File: ml/models/lstm_gpu.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import json
import time
import os

# ====== CONFIG ======
BATCH_SIZE = 128
EPOCHS = 5
HIDDEN_SIZE = 256
EMBEDDING_SIZE = 128
LEARNING_RATE = 0.001

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====== PATHS ======
train_file = r"train_numeric.json"
save_path = r"ml/models/lstm_trained_gpu.pth"
os.makedirs("ml/models", exist_ok=True)

# ====== DATASET ======
class NumericDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = torch.tensor(self.data[idx]['src'], dtype=torch.long)
        tgt = torch.tensor(self.data[idx]['tgt'], dtype=torch.long)
        return src, tgt

# ====== COLLATE FUNCTION ======
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch.to(device, non_blocking=True), tgt_batch.to(device, non_blocking=True)

# ====== VOCAB SIZE ======
def get_vocab_size(file_path):
    max_idx = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            max_idx = max(max_idx, max(item['src']), max(item['tgt']))
    return max_idx + 1

SRC_VOCAB_SIZE = get_vocab_size(train_file)
TGT_VOCAB_SIZE = SRC_VOCAB_SIZE
print(f"Detected vocab size: {SRC_VOCAB_SIZE}")

# ====== MODEL ======
class LSTMSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding_src = nn.Embedding(src_vocab_size, embed_size, padding_idx=0)
        self.embedding_tgt = nn.Embedding(tgt_vocab_size, embed_size, padding_idx=0)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, tgt_vocab_size)

    def forward(self, src, tgt):
        embedded_src = self.embedding_src(src)
        _, (hidden, cell) = self.encoder(embedded_src)
        embedded_tgt = self.embedding_tgt(tgt)
        output, _ = self.decoder(embedded_tgt, (hidden, cell))
        return self.fc(output)

# ====== TRAINING FUNCTION ======
def train_model():
    train_loader = DataLoader(
        NumericDataset(train_file),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )

    model = LSTMSeq2Seq(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting GPU training...\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        start_time = time.time()
        for src_batch, tgt_batch in train_loader:
            optimizer.zero_grad()
            output = model(src_batch, tgt_batch[:, :-1])
            loss = criterion(output.reshape(-1, TGT_VOCAB_SIZE), tgt_batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Model saved at {save_path}")

# ====== ENTRY POINT ======
if __name__ == "__main__":
    train_model()
