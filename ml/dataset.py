import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from ml.logger import get_logger

# Constants
PAD_IDX = 0  # Make sure this matches <pad> index in your vocab

# Logger
logger = get_logger("dataset.py")


class TransliterationDataset(Dataset):
    """
    PyTorch Dataset for character-level transliteration.
    Each item is a dict: {'source': [...], 'target': [...]}
    """

    def __init__(self, processed_file):
        """
        Args:
            processed_file: path to processed JSON lines (train.json or test.json)
        """
        self.data = []
        logger.info(f"Loading dataset from {processed_file}")
        try:
            with open(processed_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    self.data.append(json.loads(line))
            logger.info(f"Loaded {len(self.data)} examples from {processed_file}")
        except FileNotFoundError:
            logger.error(f"File not found: {processed_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in file {processed_file} at line {i}: {e}")
            raise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            logger.warning(f"Index {idx} out of range for dataset of size {len(self.data)}")
        sample = self.data[idx]
        src_seq = torch.tensor(sample["source"], dtype=torch.long)
        tgt_seq = torch.tensor(sample["target"], dtype=torch.long)

        # Optional debug logging for first few samples
        if idx < 3:  # Log first 3 samples for verification
            logger.debug(f"Sample {idx}: source={src_seq.tolist()}, target={tgt_seq.tolist()}")

        return {"source": src_seq, "target": tgt_seq}


def collate_fn(batch):
    """
    Pads source and target sequences to max length in the batch.
    Returns dict of tensors:
    - source: (batch_size, max_src_len)
    - target: (batch_size, max_tgt_len)
    """
    if not batch:
        logger.warning("Empty batch received in collate_fn")
        return {"source": torch.tensor([]), "target": torch.tensor([])}

    src_batch = [item["source"] for item in batch]
    tgt_batch = [item["target"] for item in batch]

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)

    logger.debug(f"Collated batch size: {len(batch)}, max_src_len: {src_batch.size(1)}, max_tgt_len: {tgt_batch.size(1)}")

    return {"source": src_batch, "target": tgt_batch}