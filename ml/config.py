import torch

# -----------------
# Data Paths
# -----------------
RAW_TRAIN = "data/raw/hin_train.json"
RAW_TEST = "data/raw/hin_test.json"
PROCESSED_TRAIN = "data/processed/train.json"
PROCESSED_TEST = "data/processed/test.json"
VOCAB_DIR = "data/vocab"
CHECKPOINT_DIR = "experiments/runs"

# -----------------
# Model Hyperparameters
# -----------------
SRC_VOCAB_SIZE = None  # will be set dynamically after loading vocab
TGT_VOCAB_SIZE = None
D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
MAX_LEN = 100

# -----------------
# Training Hyperparameters
# -----------------
BATCH_SIZE = 64
NUM_EPOCHS = 20
LR = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Special token indices (set after loading vocab)
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
