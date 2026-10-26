import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        ### 1
        vocab = []
        for st in positive:
            words = st.split(" ")
            for w in words:
                if w not in vocab:
                    vocab.append(w)
        for st in negative:
            words = st.split(" ")
            for w in words:
                if w not in vocab:
                    vocab.append(w)
        vocab.sort()
        vocab_dict = {}
        i = 1.0
        for v in vocab:
            vocab_dict[v] = i
            i += 1.0

        ### 2
        encoded_pos = []
        for st in positive:
            encoded_pos.append([])
            words = st.split(" ")
            for w in words:
                encoded_pos[-1].append(vocab_dict[w])
            encoded_pos[-1] = torch.tensor(encoded_pos[-1])
        encoded_neg = []
        for st in negative:
            encoded_neg.append([])
            words = st.split(" ")
            for w in words:
                encoded_neg[-1].append(vocab_dict[w])
            encoded_neg[-1] = torch.tensor(encoded_neg[-1])
        
        # encoded_pos = torch.tensor(encoded_pos)
        # encoded_neg = torch.tensor(encoded_neg)
        
        ### 3
        encoded = encoded_pos + encoded_neg

        ### 4
        encoded = nn.utils.rnn.pad_sequence(encoded, batch_first=True)

        return encoded

