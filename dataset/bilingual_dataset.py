import torch
from torch.utils.data import Dataset
import torch.nn as nn


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lan, tgt_lan, seq_len):
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lan = src_lan
        self.tgt_lan = tgt_lan
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        # print(src_target_pair)
        # exit(0)
        src_text = src_target_pair['translation'][self.src_lan]
        tgt_text = src_target_pair['translation'][self.tgt_lan]

        src_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        number_of_padding_tokens_src = self.seq_len - len(src_tokens) - 2
        number_of_padding_tokens_target = self.seq_len - len(tgt_tokens) - 1

        if number_of_padding_tokens_src < 0 or number_of_padding_tokens_target < 0:
            raise ValueError('Sentence is too long')

        encoder_input = torch.cat([self.sos_token,
                                   torch.tensor(src_tokens, dtype=torch.int64),
                                   self.eos_token,
                                   torch.tensor([self.pad_token] * number_of_padding_tokens_src, dtype=torch.int64)
                                   ], dim=0)

        decoder_input = torch.cat([self.sos_token,
                                   torch.tensor(tgt_tokens, dtype=torch.int64),
                                   torch.tensor([self.pad_token] * number_of_padding_tokens_target, dtype=torch.int64)
                                   ], dim=0)

        label = torch.cat([torch.tensor(tgt_tokens, dtype=torch.int64),
                           self.eos_token,
                           torch.tensor([self.pad_token] * number_of_padding_tokens_target, dtype=torch.int64)
                           ], dim=0)

        assert encoder_input.size(0) == self.seq_len, 'batch dimension needs to be added'
        assert decoder_input.size(0) == self.seq_len, 'batch dimension needs to be added'
        assert label.size(0) == self.seq_len, 'batch dimension needs to be added'

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & mask(
                decoder_input.shape[0]),
            "src_txt": src_text,
            "tgt_txt": tgt_text,
        }


def mask(size):
    up_triangular = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return up_triangular == 0
