import numpy as np

def rescale01(X, M):
    """
    Rescale the data to the range [0, 1].

    Parameters:
    X (numpy.ndarray): The data array. (N x T)
    M (numpy.ndarray): The mask array. (N x T)

    Returns:
    numpy.ndarray: The rescaled data array.
    """
    minX = X[:, ~M].min()#, keepdims=True)
    maxX = X[:, ~M].max()#, keepdims=True)
    X[:, ~M] = (X[:, ~M] - minX) / (maxX - minX + 1e-8)
    X[:, M] = 0
    return X.astype(np.float32)

def temporal_flip_concatenate(X, M):
    """
    Temporally flip and concatenate the input data and mask.
    """
    X_pos = np.clip(X, a_min=0, a_max=None)  # retains only positive values
    X_neg = np.clip(X, a_min=None, a_max=0)  # retains only negative values
    return np.concatenate([X_pos, -X_neg], axis=1), np.concatenate([M, M], axis=1)

def unit_flip_concatenate(X, M):
    """
    Unit flip and concatenate the input data and mask.
    """
    X_pos = np.clip(X, a_min=0, a_max=None)  # retains only positive values
    X_neg = np.clip(X, a_min=None, a_max=0)  # retains only negative values
    return np.concatenate([X_pos, -X_neg], axis=0), np.concatenate([M, M], axis=0)
