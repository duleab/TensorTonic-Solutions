import numpy as np

def matrix_transpose(A):
    # Ensure input is a numpy array to use .shape
    A = np.array(A)
    rows, cols = A.shape
    # Initialize an empty matrix of shape (cols, rows)
    result = np.zeros((cols, rows), dtype=A.dtype)
    
    for i in range(rows):
        for j in range(cols):
            # Map element at (i, j) to (j, i)
            result[j, i] = A[i, j]
            
    return result