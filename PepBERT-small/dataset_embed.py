import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random

class PeptideDataset(Dataset):

    def __init__(self, ds, tokenizer,  seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer = tokenizer

        self.sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
        self.mask_token = torch.tensor([tokenizer.token_to_id("[MASK]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        seq = self.ds[idx]
        masked_seq_ids, origi_seq_ids = random_word(seq, self.tokenizer)
        num_padding_tokens = self.seq_len - len(masked_seq_ids) - 2  # We will add <s> and </s>
        if num_padding_tokens < 0:
            raise ValueError("Sentence is too long")
        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(masked_seq_ids, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        label = torch.cat(
            [
                self.sos_token,
                torch.tensor(origi_seq_ids, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len, "encoder_input size is not matched"
        assert label.size(0) == self.seq_len, "label size is not matched"

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "label": label
        }

def random_word(sentence,tokenizer):        
    origi_seq_ids = tokenizer.encode(sentence).ids # this is a list output, each element is a int
    masked_seq_ids = origi_seq_ids
    assert len(masked_seq_ids) == len(origi_seq_ids) , "masked seq has difference length as the original seq"
    return  masked_seq_ids, origi_seq_ids
