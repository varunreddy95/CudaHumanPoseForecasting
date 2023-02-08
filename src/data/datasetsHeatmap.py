from torch.utils.data import Dataset
from src.configs.config import cfg
import pickle
import os
import random
import numpy as np
import torch
from paths import H36_DIR
import time
from torch.utils.data import DataLoader
import scipy.stats as st

class HposeHeat(Dataset):
    def __init__(self, subset="train", normalize=True) -> None:
        self.sigma = 0.5
        self.heatmap_sz = 64
        self.num_frames = 10
        if subset == "valid":
            with open(os.path.join(H36_DIR, "h36m_validation_processed_all_sequence.pkl"), "rb") as file:
                self.data = pickle.load(file)

        elif subset == "train":
            with open(os.path.join(H36_DIR, "h36m_train_processed_all_sequence.pkl"), "rb") as file:
                self.data = pickle.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        selected_frames = self.data[idx]
        input_poses = torch.stack([torch.flatten(torch.tensor(i['joints_2d']))
                                   for i in selected_frames[:10]])

        ground_truth = torch.stack([torch.flatten(torch.tensor(i['joints_2d']))
                                    for i in selected_frames[10:]])


        input_heat = self.create_heatmap(input_poses)
        gt_heat = self.create_heatmap(ground_truth)
        # Do we need normalization for heatmap?
        # if normalize:
        #     ground_truth = torch.div(ground_truth, 1000.0)
        #     input_poses = torch.div(input_poses, 1000.0)

        return input_heat, gt_heat

    # Code modified from https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/dataset/JointsDataset.py
    def create_heatmap(self, frames):
        stride = 1000 // self.heatmap_sz
        range_heat = self.sigma * 8
        heatmap = np.zeros((self.num_frames, 17, self.heatmap_sz, self.heatmap_sz), dtype=np.float32)
        for f_idx in range(self.num_frames):
            joints = frames[f_idx]
            joints = joints.view(17,2)    
            for idx, points in enumerate(joints):
                # convert points position in image into heatmap 64 * 64
                #print(joints)
                hm_x = int(joints[idx][0] / stride + 0.5)
                hm_y = int(joints[idx][1] / stride + 0.5)

                # address the boundary problem
                x_low = int(max(0, hm_x - range_heat))
                y_low = int(max(0, hm_y - range_heat))
                x_high = int(min(self.heatmap_sz, hm_x + range_heat + 1))
                y_high = int(min(self.heatmap_sz, hm_y + range_heat + 1))

                sz = int(2 * range_heat + 1)  # sz = int(2 * range_heat + 1)
                # x = np.arange(0, sz, 1, np.float32)      # 9
                # y = x[:, np.newaxis]                     # 9, 1
                # x0 = y0 = sz // 2
                # g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (self.sigma ** 2)))       # sz * sz (9*9)
                temp_x = np.linspace(-4.5, 4.5, 9 + 1)
                kern1d = np.diff(st.norm.cdf(temp_x))
                g = np.outer(kern1d, kern1d)
                g = g / g.max()

                # Normalize the gaussian
                # g_norm = (g - np.min(g)) / (np.max(g) - np.min(g))

                # address the boundary problem for gaussian tensor
                g_x_low = int(max(0, range_heat - hm_x))
                g_x_high = int(min(self.heatmap_sz, (hm_x + range_heat + 1)) - (hm_x - range_heat))
                g_x_high = max(g_x_high, 0)
                g_y_low = int(max(0, range_heat - hm_y))
                g_y_high = int(min(self.heatmap_sz, (hm_y + range_heat + 1)) - (hm_y - range_heat))
                g_y_high = max(g_y_high, 0)

                heatmap[f_idx][idx][y_low:y_high, x_low:x_high] = g[g_y_low:g_y_high, g_x_low:g_x_high]
                # except ValueError as e:
                #     print(e)
                #     print("y_low {} y_high {} x_low {} x_high {}".format(y_low, y_high, x_low, x_high))
                #     print("g_y_low {} g_y_high {} g_x_low {} g_x_high {}".format(g_y_low, g_y_high, g_x_low, g_x_high))
                #     continue

        return torch.from_numpy(heatmap)


if __name__ == "__main__":
    validation_dataset = HposeHeat(subset="train")
    dataloader = DataLoader(validation_dataset, batch_size=16)
    input, gt = next(iter(dataloader))