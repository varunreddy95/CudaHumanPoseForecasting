import torch.nn as nn
import torch
from torch.nn import MSELoss

from src.data.datasets import Hpose
from torch.utils.data import DataLoader
import numpy as np

class MPJPE(nn.Module):
    def __init__(self) -> None:
        super(MPJPE, self).__init__()

    def forward(self, output, target):
        return torch.mean(torch.sqrt((output-target)**2))


class L1Custom(nn.Module):
    def __init__(self) -> None:
        super(L1Custom, self).__init__()
        self.L1 = nn.L1Loss(reduction="none")

    def forward(self, output, target):
        loss = self.L1(output, target)
        loss = torch.sum(loss, dim=2)
        loss = torch.sum(loss, dim=0)
        loss = torch.mul(loss, torch.tensor([1,1,2,2,3,3,4,4,5,5], device=output.device))
        return torch.sum(loss)


def pck(gt, pred, th):
    '''
    :param gt: ground truth should have shape of [BSZ Frames 34] or [BSZ Frames 17 hm_sz hm_sz]
    :param pred: same as gt
    :param th: Threshold
    :return:
    '''
    n_frames = 10
    correct = 0
    bsz = gt.size(0)
    total_correct = n_frames * 17 * bsz
    for x in range(bsz):
        for f in range(n_frames):              # loop all frames
            for joint in range(0, 34, 2):
                dist = (gt[x][f][joint] - pred[x][f][joint])**2 + (gt[x][f][joint+1] - pred[x][f][joint+1])**2

                dist = np.sqrt(dist)
                if dist < th: correct += 1

    return correct / total_correct


if __name__ == "__main__":

    dataset = Hpose()
    trainloader = DataLoader(dataset, batch_size=2)

    seed, gt = next(iter(trainloader))

    loss = L1Custom()

    print(loss(seed, gt))

    mpjpe = MPJPE()

    print(mpjpe(seed, gt))
