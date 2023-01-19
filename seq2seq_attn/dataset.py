from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from seq2seq_attn.constants import device

SOS_token = 0
EOS_token = 1


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(" ")]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return TwoLangs(enc=input_tensor, dec=target_tensor)


@dataclass
class TwoLangs:
    enc: torch.tensor
    dec: torch.tensor


class CustomDs(Dataset):
    def __init__(self, input_lang, output_lang, pairs):
        self.training_pairs = [
            tensorsFromPair(input_lang, output_lang, pair) for pair in pairs
        ]

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, idx):
        return self.training_pairs[idx]
