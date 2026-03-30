import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    if not seqs:
        return np.zeros((0, 0), dtype=int)
    if max_len is None:
        max_len = max(len(seq) for seq in seqs) if seqs else 0
    num_seqs = len(seqs)
    result = np.full((num_seqs, max_len), pad_value, dtype=int)
    for i, seq in enumerate(seqs):
        copy_len = min(len(seq), max_len)
        if copy_len > 0:
            result[i, :copy_len] = seq[:copy_len]
    return result