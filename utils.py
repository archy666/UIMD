from sklearn.preprocessing import OneHotEncoder
import numpy as np
from numpy.random import randint
import torch
import random
from scipy.spatial.distance import pdist, squareform
from typing import Union



def get_sn(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num: view number
    :param alldata_len: number of samples
    :param missing_rate: Defined in the paper
    :return:Sn
    """
    one_rate = 1-missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    return matrix

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)  # Python built-in random number generator
    np.random.seed(seed)  # Numpy's random number generator
    torch.manual_seed(seed)  # PyTorch's random number generator
    torch.cuda.manual_seed(seed)  # Random number generator for GPU when used
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility, but may decrease speed
    torch.backends.cudnn.benchmark = False  # Turn off automatic algorithm search for stability in experiments


def Normalize(data):
    """
    :param data:Input data
    :return:normalized data
    """
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - m) / (mx - mn)

def pairwise_distances(x):
    bn = x.shape[0]
    x = x.view(bn, -1)
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def calculate_gram_mat(x, sigma):
    dist = pairwise_distances(x)
    return torch.exp(-dist / sigma)


def reyi_entropy(x, sigma):
    alpha = 1.01#1.01
    k = calculate_gram_mat(x, sigma)
    k = k / torch.trace(k)
    # eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    try:
        eigv = torch.abs(torch.linalg.eigh(k)[0])
    except:
        eigv = torch.diag(torch.eye(k.shape[0]))
    # eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x, y, s_x, s_y):
    alpha = 1.01#1.01
    x = calculate_gram_mat(x, s_x)
    y = calculate_gram_mat(y, s_y)
    k = torch.mul(x, y)
    k = k / torch.trace(k)
    # eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eigv = torch.abs(torch.linalg.eigh(k)[0])

    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def calculate_MI(x, y, s_x, s_y):
    Hx = reyi_entropy(x, sigma=s_x)
    Hy = reyi_entropy(y, sigma=s_y)
    Hxy = joint_entropy(x, y, s_x, s_y)
    Ixy = Hx + Hy - Hxy
    return Ixy

def get_kernelsize(features: torch.Tensor, selected_param: Union[int, float]=0.15, select_type: str='meadian'):
    ### estimating kernelsize with data with the rule-of-thumb
    features = torch.flatten(features, 1).cpu().detach().numpy()
    k_features = squareform(pdist(features))
    if select_type=='min':
        kernelsize = np.sort(k_features, 1)[:, :int(selected_param)].mean()
    elif select_type=='max':
        kernelsize = np.sort(k_features, 1)[:, int(selected_param):].mean()
    elif select_type=='meadian':
        triu_indices = np.triu_indices(k_features.shape[0], 1)
        kernelsize = selected_param*np.median(k_features[triu_indices])
        
    else:
        kernelsize = 1.0
    # if kernelsize<EPSILON:
    #     kernelsize = torch.tensor(EPSILON, device=features.device)
    return kernelsize

def hsic(x, y, s_x, s_y):
    s_x = s_x or 0.3
    s_y = s_y or 0.3        
    m = x.shape[0]  
    K = calculate_gram_mat(x, sigma=s_x)
    L = calculate_gram_mat(y, sigma=s_y)
    H = torch.eye(m) - 1.0/m * torch.ones((m, m))
    H = H.float().to(x.device)
    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H))))/((m-1)**2)
    return HSIC



def human_bytes(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"
