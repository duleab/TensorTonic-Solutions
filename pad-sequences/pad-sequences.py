import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    if not seqs:
        return np.zeros((0, max_len or 0))
    
    n = len(seqs)
    actual_max = max(len(seq) for seq in seqs)
    l = max_len if max_len is not None else actual_max
    
    result = np.full((n, l), pad_value, dtype=np.int32)
    
    for i, seq in enumerate(seqs):
        trunc_seq = seq[:l]
        result[i, :len(trunc_seq)] = trunc_seq
        
    return result