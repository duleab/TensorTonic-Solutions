import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.word_to_id = {token: i for i, token in enumerate(special_tokens)}
        self.id_to_word = {i: token for i, token in enumerate(special_tokens)}
        self.vocab_size = len(special_tokens)
        for text in texts:
            for word in text.split():
                if word not in self.word_to_id:
                    self.word_to_id[word] = self.vocab_size
                    self.id_to_word[self.vocab_size] = word
                    self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        return [self.word_to_id.get(word, self.word_to_id[self.unk_token]) for word in text.split()]

    def decode(self, ids: List[int]) -> str:
        return " ".join([self.id_to_word.get(idx, self.unk_token) for idx in ids])