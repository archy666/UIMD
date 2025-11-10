from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from utils import get_sn
import scipy.io as sio
import torch
from utils import set_seed
from sklearn.model_selection import train_test_split


set_seed(123)


class import_data(Dataset):
    def __init__(self, data_path, missing_rate, random_state, train=True):
        self.views, self.Sn_s, self.labels = prepare_multiview_data(
            data_path, missing_rate, random_state, train
        )

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        views = [torch.tensor(view[idx], dtype=torch.float32) for view in self.views]
        Sn = [torch.tensor(Sn[idx], dtype=torch.float32) for Sn in self.Sn_s]
        label = self.labels[idx]
        return {'views': views, 'Sn': Sn, 'label': label}


def prepare_multiview_data(data_path, missing_rate, random_state, train):
    temp = sio.loadmat(data_path)
    label_temp = torch.tensor(np.squeeze(temp['Y']) - 1).long() if np.min(temp['Y']) == 1 else torch.tensor(np.squeeze(temp['Y'])).long()
    view_num = temp['X'][0].shape[0]
    alldata_len = len(label_temp)
    Sn = get_sn(view_num, alldata_len, missing_rate)
    scaler = MinMaxScaler()
    processed_data = []
    for i in range(view_num):
        data_i = temp['X'][0][i]
        scaled_data = scaler.fit_transform(data_i)
        mask = Sn[:, i].reshape(alldata_len, 1)
        masked_data = np.multiply(scaled_data, mask)
        processed_data.append(masked_data)
    _, _, train_indices, test_indices = train_test_split(
        label_temp, range(alldata_len), test_size=0.2, stratify=label_temp, random_state=random_state
    )

    processed_data = [np.array(view_data) for view_data in processed_data]
    Sn = np.array(Sn)

    if train:
        indices = train_indices
    else:
        indices = test_indices

    views = [data[indices] for data in processed_data]
    Sn_selected = [Sn[indices, i].reshape(-1, 1) for i in range(view_num)]
    labels = label_temp[indices]

    return views, Sn_selected, labels