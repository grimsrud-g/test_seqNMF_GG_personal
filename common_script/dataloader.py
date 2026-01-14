import numpy as np
from torch.utils.data import Dataset
import random

class IndividualDataset(Dataset):
    def __init__(self, data_dir, size, subject, L):
        self.data_dir = data_dir
        self.size=size         
        self.subject = subject
        # goal here is to do indexing internally
        #self.subjects = list(subject_list)
        #self.size = len(self.subjects)
        self.L = L

    def __len__(self):
        return self.size 

    def __getitem__(self):
        X = np.load(f'{self.data_dir}/{self.subject}_X.npy')
        M = np.load(f'{self.data_dir}/{self.subject}_M.npy')
        Xpad = np.pad(X, ((0,0), (self.L,self.L))).astype(np.float32)
        Mpad = np.pad(M, ((0,0), (self.L,self.L))).astype(np.bool_)

        return Xpad, Mpad
    
class AllSubjectsDataset(Dataset):
    def __init__(self, data_dir, size, subject_list, L, K, seed):
        self.data_dir = data_dir
        self.size = size
        # copy of subj list
        inSubjs=subject_list.copy()
        # randomize which subjects we are taking w/r/t K so same PTs arent used across Ks
        # also shuffling based on which seed, so same PTs are not used across seeds
        rnd = random.Random((int(K) << 32) ^ int(seed))
        rnd.shuffle(inSubjs)
        # truncate subject list to length of subjects to include in this particular iteration
        inSubjs = inSubjs[:size]
        self.subject = inSubjs
        # subjects add
        self.subjects = list(inSubjs)
        self.L = L

    def __len__(self):
        return self.size 

    def __getitem__(self,index):
        #subject = self.subject
        # also with goal of doing indexing internally
        subject = self.subjects[index]
        X = np.load(f'{self.data_dir}/{subject}_X.npy').astype(np.float32)
        M = np.load(f'{self.data_dir}/{subject}_M.npy')
	# moved padding to fit script to apply to every time series and not already-concatenated time series
        #Xpad = np.pad(X, ((0,0), (self.L,self.L))).astype(np.float32)
        #Mpad = np.pad(M, ((0,0), (self.L,self.L))).astype(np.bool_)

        return X, M, index
