from locale import normalize
from torch.utils.data import Dataset
from src.configs.config import cfg
import paths
import pickle
import os
import random
import torch.nn.functional as F
import torch


class Hpose(Dataset):
    def __init__(self, subset="train", normalize=True) -> None:
        if subset == "valid":
            with open(os.path.join(paths.H36_DIR, "h36m_validation_processed_all_sequence.pkl"), "rb") as file:
                self.data = pickle.load(file)

        elif subset == "train":
            with open(os.path.join(paths.H36_DIR, "h36m_train_processed_all_sequence.pkl"), "rb") as file:
                self.data = pickle.load(file)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        selected_frames = self.data[idx]
        input_poses = torch.stack([torch.flatten(torch.tensor(i['joints_2d']))
                                   for i in selected_frames[:10]])

        ground_truth = torch.stack([torch.flatten(torch.tensor(i['joints_2d']))
                                    for i in selected_frames[10:]])

        if normalize:
            ground_truth = torch.div(ground_truth, 1000.0)
            input_poses = torch.div(input_poses, 1000.0)

        return input_poses, ground_truth

