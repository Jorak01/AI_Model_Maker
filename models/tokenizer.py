"""Simple word/char tokenizer for conversational AI."""

import pickle
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter


class Tokenizer:
    PAD, UNK, BOS, EOS, SEP = "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"

    def __init__(self, vocab_size: int = 10000, method: str = "word"):
        self.vocab_size = vocab_size
        self.method = method

        self.token2id: Dict[str, int] = {
            self.PAD: 0, self.UNK: 1, self.BOS: 2, self.EOS: 3, self.SEP: 4
        }
        self.id2token: Dict[int, str] = {v: k for k, v in self.token2id.items()}

        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.sep_token_id = 4

    def _clean(self, text: str) -> str:
        return " ".join(text.lower().split())

    def _tokenize(self, text: str) -> List[str]:
        if self.method == "word":
            return re.findall(r'\w+|[^\w\s]', text)
        return list(text)

    def build_vocab(self, texts: List[str]):
        tokens = []
        for t in texts:
            tokens.extend(self._tokenize(self._clean(t)))
        for token, _ in Counter(tokens).most_common(self.vocab_size - len(self.token2id)):
            if token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx] = token
        print(f"Vocab built: {len(self.token2id)} tokens")

    def encode(self, text: str, add_special: bool = True, max_length: Optional[int] = None, pad: bool = False) -> List[int]:
        ids = [self.token2id.get(t, self.unk_token_id) for t in self._tokenize(self._clean(text))]
        if add_special:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        if max_length:
            ids = ids[:max_length]
            if pad:
                ids += [self.pad_token_id] * (max_length - len(ids))
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        special = {0, 2, 3, 4} if skip_special else set()
        tokens = [self.id2token.get(i, self.UNK) for i in ids if i not in special]
        if self.method == "word":
            result = ""
            for i, tok in enumerate(tokens):
                if i == 0 or tok in ".,!?;:')]}":
                    result += tok
                elif i > 0 and tokens[i - 1] in "([{":
                    result += tok
                else:
                    result += " " + tok
            return result
        return "".join(tokens)

    def encode_conversation(self, prompt: str, response: str, max_length: int = 128) -> Tuple[List[int], List[int]]:
        p_ids = self.encode(prompt, add_special=False)
        r_ids = self.encode(response, add_special=False)
        ids = [self.bos_token_id] + p_ids + [self.sep_token_id] + r_ids + [self.eos_token_id]
        ids = ids[:max_length]
        target = ids.copy()
        pad_len = max_length - len(ids)
        ids += [self.pad_token_id] * pad_len
        target += [self.pad_token_id] * pad_len
        return ids, target

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'vocab_size': self.vocab_size, 'method': self.method,
                         'token2id': self.token2id, 'id2token': self.id2token}, f)

    @classmethod
    def load(cls, path: str) -> 'Tokenizer':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        tok = cls(data['vocab_size'], data['method'])
        tok.token2id = data['token2id']
        tok.id2token = data['id2token']
        return tok

    def __len__(self):
        return len(self.token2id)
