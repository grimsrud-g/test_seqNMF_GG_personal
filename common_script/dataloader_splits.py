# dataloader_splits.py
import os
import numpy as np
from torch.utils.data import Dataset
import random

class NPYPairsDataset(Dataset):
    """
    Minimal dataset that loads explicit (x_path, m_path) pairs.
    This avoids any guessing about how files are named or where they live.
    """
    def __init__(self, pairs, pad_L=0, dtype_X=np.float32):
        """
        pairs: list of dicts or tuples with keys/fields:
               - 'x': path to X .npy
               - 'm': path to M .npy
               - optional 'tag': string label for logging
        pad_L: how many frames to pad on both sides (0 means no padding)
        """
        self.pairs = []
        for p in pairs:
            if isinstance(p, dict):
                x_path, m_path = p["x"], p["m"]
                tag = p.get("tag", os.path.basename(x_path))
            else:
                # tuple or list
                x_path, m_path = p[0], p[1]
                tag = p[2] if len(p) > 2 else os.path.basename(x_path)
            if not (os.path.exists(x_path) and os.path.exists(m_path)):
                # skip missing pairs
                continue
            self.pairs.append({"x": x_path, "m": m_path, "tag": tag})
        self.pad_L = int(pad_L) if pad_L is not None else 0
        self.dtype_X = dtype_X

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        x_path = self.pairs[index]["x"]
        m_path = self.pairs[index]["m"]

        X = np.load(x_path)
        M = np.load(m_path)

        if self.pad_L > 0:
            L = self.pad_L
            X = np.pad(X, ((0, 0), (L, L))).astype(self.dtype_X)
            M = np.pad(M, ((0, 0), (L, L))).astype(np.bool_)

        # return optional tag for bookkeeping
        return X.astype(self.dtype_X), M.astype(np.bool_), index

class AllSubjectsDataset(Dataset):
    """
    Backward-compatible variant that still works off 'subject labels' if you want it.
    It expects files named {subject}_X.npy / {subject}_M.npy under data_dir.
    """
    def __init__(self, data_dir, size, subject_list, L, K, seed):
        self.data_dir = data_dir
        self.size = size
        inSubjs = subject_list.copy()
        rnd = random.Random((int(K) << 32) ^ int(seed))
        rnd.shuffle(inSubjs)
        self.subjects = inSubjs[:size]
        self.L = int(L)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        subject = self.subjects[index]
        x_path = os.path.join(self.data_dir, f"{subject}_X.npy")
        m_path = os.path.join(self.data_dir, f"{subject}_M.npy")
        X = np.load(x_path)
        M = np.load(m_path)
        # Padding is often done earlier; leave raw here to avoid double-padding
        return X.astype(np.float32), M.astype(np.bool_), index

class IndividualDataset(Dataset):
    """
    Single subject file pair loader (legacy but fixed to accept an index).
    """
    def __init__(self, data_dir, size, subject, L):
        self.data_dir = data_dir
        self.size = size
        self.subject = subject
        self.L = int(L)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        x_path = os.path.join(self.data_dir, f"{self.subject}_X.npy")
        m_path = os.path.join(self.data_dir, f"{self.subject}_M.npy")
        X = np.load(x_path)
        M = np.load(m_path)
        if self.L > 0:
            X = np.pad(X, ((0,0), (self.L,self.L))).astype(np.float32)
            M = np.pad(M, ((0,0), (self.L,self.L))).astype(np.bool_)
        return X, M

