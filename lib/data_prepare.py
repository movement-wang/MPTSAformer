import os
import numpy
import torch
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader

from .utils import StandardScaler


class PretrainDataset(Dataset):
    def __init__(self, data: torch.Tensor, index: numpy.ndarray):
        super().__init__()
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index: int):
        item = list(self.index[index])
        long_history_data = self.data[item[0] : item[1]]
        return long_history_data


class TrainDataset(Dataset):
    def __init__(self, data: torch.Tensor, index: numpy.ndarray, pretrain_length: int):
        super().__init__()
        self.data = data
        self.index = index
        self.pretrain_length = pretrain_length
        # self.scaler = scaler


    def __len__(self):
        return len(self.index)

    def __getitem__(self, index: int):
        item = list(self.index[index])
        history_data = self.data[item[0] : item[1]]
        future_data = self.data[item[1] : item[2], :, :1]
        if item[1] - self.pretrain_length < 0:
            long_history_data = torch.cat(
                [torch.zeros(self.pretrain_length - item[1], self.data.shape[1], self.data.shape[2]), self.data[: item[1]]],
                dim=0,
            )
        else:
            long_history_data = self.data[item[1] - self.pretrain_length : item[1]]

        # history_data = deepcopy(history_data)
        # history_data[..., 0] = self.scaler.transform(history_data[..., 0])
        # long_history_data = deepcopy(long_history_data)
        # long_history_data[..., 0] = self.scaler.transform(long_history_data[..., 0])

        return history_data, future_data, long_history_data


def get_train_dataloader(data_dir, tod=True, dow=True, batch_size=16, in_steps=12, out_steps=12, pretrain_length=864):
    data = numpy.load(os.path.join(data_dir, "data.npz"))["data"].astype(numpy.float32)
    features = [0]
    if tod:
        features.append(1)
    if dow:
        features.append(2)
    processed_data = data[..., features]

    index = numpy.load(os.path.join(data_dir, f"index_{in_steps}_{out_steps}.npz"))
    train_index = index["train"]  # (num_samples, 3)
    val_index = index["val"]
    test_index = index["test"]

    train_data = processed_data[: train_index[-1, 1], ..., 0]
    scaler = StandardScaler(mean=train_data.mean(), std=train_data.std())

    trainset = TrainDataset(torch.FloatTensor(processed_data), train_index, pretrain_length=pretrain_length)
    valset = TrainDataset(torch.FloatTensor(processed_data), val_index, pretrain_length=pretrain_length)
    testset = TrainDataset(torch.FloatTensor(processed_data), test_index, pretrain_length=pretrain_length)

    trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valset_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    testset_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainset_loader, valset_loader, testset_loader, scaler


def get_pretrain_dataloader(data_dir, tod=True, dow=True, batch_size=16, steps=864):
    data = numpy.load(os.path.join(data_dir, "data.npz"))["data"].astype(numpy.float32)
    features = [0]
    if tod:
        features.append(1)
    if dow:
        features.append(2)
    processed_data = data[..., features]

    index = numpy.load(os.path.join(data_dir, f"index_{steps}.npz"))
    train_index = index["train"]  # (num_samples, 3)
    val_index = index["val"]

    train_data = processed_data[: train_index[-1, 1], ..., 0]
    scaler = StandardScaler(mean=train_data.mean(), std=train_data.std())

    trainset = PretrainDataset(torch.FloatTensor(processed_data), train_index)
    valset = PretrainDataset(torch.FloatTensor(processed_data), val_index)

    trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valset_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    return trainset_loader, valset_loader, scaler
