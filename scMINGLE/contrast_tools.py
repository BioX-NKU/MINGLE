import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def balance_sampling(X, y, target_samples=300):
    """
    Using different sample strategy to balance cell type number to a fixed value

    Parameters
    ----------
    X: numpy matrix
        numpy data matrix

    y: numpy array
        labels that corresponding the data

    target_samples: int, optional (default=300)
        the target fixed number of cells in each cell type

    Returns
    ----------
    real_X_matrix: numpy matrix
        sample matrix that exists in real dataset.

    real_y_vector: numpy array
        sample labels that exists in real dataset.

    noise_X_matrix: numpy matrix
        sample matrix that oversampled.

    noise_y_vector: numpy array
        sample labels that corresponding the noise.
    """

    unique_classes, counts = np.unique(y, return_counts=True)
    real_X, real_y = [], []
    noise_X, noise_y = [], []

    if counts.max() < target_samples:
        target_samples = counts.max()

    for class_val in unique_classes:
        class_indices = np.where(y == class_val)[0]
        class_samples = X[class_indices]
        class_labels = y[class_indices]

        # Undersampling
        if len(class_indices) >= target_samples:
            sampled_indices = np.random.choice(class_indices, target_samples, replace=False)
            real_X.append(X[sampled_indices])
            real_y.append(y[sampled_indices])
        # Oversampling
        else:
            real_X.append(class_samples)
            real_y.append(class_labels)
            samples_to_generate = target_samples - len(class_indices)
            for _ in range(samples_to_generate):
                sample_index = np.random.choice(class_indices)
                sample = X[sample_index].copy()
                noise_X.append(sample)
                noise_y.append(np.array([class_val]))

    real_X_matrix, real_y_vector = np.vstack(real_X), np.hstack(real_y)
    try:
        noise_X_matrix, noise_y_vector = np.vstack(noise_X), np.hstack(noise_y)
    except:
        noise_X_matrix, noise_y_vector = noise_X, noise_y
             
    return real_X_matrix, real_y_vector, noise_X_matrix, noise_y_vector

def pair_generating(X, y, seed=42):
    """
    Generate sample pairs

    Parameters
    ----------
    X: numpy matrix
        numpy data matrix

    y: numpy array
        labels that corresponding to the data

    target_samples: int
        the target fixed number of cells in each cell type

    Returns
    ----------
    positive_pairs: list 
    negative_pairs: list

    """

    positive_pairs = []
    negative_pairs = []

    for i in range(len(X)):
        # positive pair
        same_class_indices = np.where(y == y[i])[0]
        same_class_indices = same_class_indices[same_class_indices != i]
        if len(same_class_indices) > 0:
            positive_pair_index = np.random.choice(same_class_indices)
            positive_pairs.append((X[i], X[positive_pair_index]))

        # negative pair
        diff_class_indices = np.where(y != y[i])[0]
        if len(diff_class_indices) > 0:
            negative_pair_index = np.random.choice(diff_class_indices)
            negative_pairs.append((X[i], X[negative_pair_index]))

    min_pairs = min(len(positive_pairs), len(negative_pairs))
    positive_pairs = shuffle(positive_pairs, random_state=seed)[:min_pairs]
    negative_pairs = shuffle(negative_pairs, random_state=seed)[:min_pairs]

    return positive_pairs, negative_pairs

class CL(nn.Module):
    def __init__(self, inputs, hidden_num, outputs, device):
        """
        multiple layers perceptron with specific inputs dimension, hidden dimension, output dimension.
        
        Parameters
        ----------
        inputs: int
            input dimension.
            
        hidden_num: int
            input dimension.
        
        outputs: int
            output dimension.
            
        """

        super(CL, self).__init__()
        self.device = device
        self.outputs=outputs
        self.Linear1=nn.Linear(inputs, hidden_num).to(self.device)
        self.Linear2=nn.Linear(hidden_num, outputs).to(self.device)
        self.scs=nn.CosineSimilarity(dim=-1)
        self.relu = nn.ReLU()
        
        
    def forward(self, x1, x2):
        x1 = x1.float().to(self.device)
        x2 = x2.float().to(self.device)
        x1 = 0.1*F.normalize(nn.Sigmoid()(self.Linear1(x1)).to(self.device))
        x2 = 0.1*F.normalize(nn.Sigmoid()(self.Linear1(x2)).to(self.device))
        x1 = self.Linear2(x1)
        x2 = self.Linear2(x2)
        cosine=self.scs(x1, x2)
        return cosine
        
    def get_embedding(self, x):
        x = 0.1*F.normalize(nn.Sigmoid()(self.Linear1(x)).to(self.device))
        x = (self.Linear2(x))
        return x
    
class ContrastiveDataset(Dataset):
    def __init__(self, positive_pairs, negative_pairs):
        self.pairs = positive_pairs + negative_pairs
        self.labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (x1, x2), label = self.pairs[idx], self.labels[idx]
        return x1, x2, label

def mask(noisedata,rate):
    """
    Mask the dulplicated samples to make noise.
    
    Parameters
    ----------
    noisedata: AnnData
        the data for processing.
        
    rate: float
        masking rate.
    
    outputs:
        processed data.

    Returns
    ----------
    traindata_copy: AnnData
        data processed by masking strategy    
    """
    traindata_copy = noisedata.copy()
    non_zero = np.array(np.where(traindata_copy != 0))
    index_list = random.sample(list(range(non_zero.shape[1])), int(len(list(range(non_zero.shape[1])))*rate))
    mask_index_0 = non_zero[:,index_list]
    mask_index = tuple(mask_index_0.tolist())
    traindata_copy[mask_index] =0
    return traindata_copy