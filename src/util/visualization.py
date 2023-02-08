import torch
import numpy as np
from torchvision.utils import draw_keypoints
import matplotlib.pyplot as plt


def draw_keypoint_sequence(seed_frames: torch.Tensor, ground_truth: torch.Tensor, prediction: torch.Tensor, sequence_number=0):
    H36M_SKELETON = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                  [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
    seed_images = []
    gt_images = []
    prediction_images = []
    for i in range(10):
        #print(seed_frames[0,i,:].unsqueeze(0).view(-1,17,2).size())
        reference_image = torch.zeros((3,1000,1000)).type(torch.uint8)
        kpt_img = draw_keypoints(
            image=reference_image,
            keypoints=seed_frames[sequence_number,i,:].unsqueeze(0).view(-1,17,2),
            colors="red",
            radius=10,
            width=8,
            connectivity=H36M_SKELETON
        )
        seed_images.append(kpt_img.unsqueeze(0))
        reference_image = torch.zeros((3,1000,1000)).type(torch.uint8)
        kpt_img = draw_keypoints(
            image=reference_image,
            keypoints=ground_truth[sequence_number,i,:].unsqueeze(0).view(-1,17,2),
            colors="red",
            radius=10,
            width=8,
            connectivity=H36M_SKELETON
        )
        gt_images.append(kpt_img.unsqueeze(0))
        reference_image = torch.zeros((3,1000,1000)).type(torch.uint8)
        kpt_img = draw_keypoints(
            image=reference_image,
            keypoints=prediction[sequence_number,i,:].unsqueeze(0).view(-1,17,2),
            colors="red",
            radius=10,
            width=8,
            connectivity=H36M_SKELETON
        )
        prediction_images.append(kpt_img.unsqueeze(0))
    return torch.cat(seed_images), torch.cat(gt_images), torch.cat(prediction_images)

def draw_heatmap(seed_frames: torch.Tensor, ground_truth: torch.Tensor, prediction: torch.Tensor):
    seed_images = draw_single_heat(seed_frames)
    gt_images = draw_single_heat(ground_truth)
    prediction_images = draw_single_heat(prediction.detach().cpu())
    return torch.cat(seed_images), torch.cat(gt_images), torch.cat(prediction_images)


def draw_single_heat(input_frame):
    fig = plt.figure(figsize=(15, 15))
    for f in range(10):
        reference_image = np.zeros((64, 64))
        for i in range(17):
            reference_image = np.add(reference_image, input_frame[1][f][i].numpy())
        fig.add_subplot(1, 10, f + 1)
        plt.imshow(reference_image)
    plt.show()

def draw_grid_image(whole, columns, rows):
    fig = plt.figure(figsize=(17, 9))


    # ax enables access to manipulate each of subplots
    ax = []

    for i in range(columns*rows):
        img = torch.permute(whole[i], (1,2,0)).numpy()
        ax.append( fig.add_subplot(rows, columns, i+1) )
        plt.axis('off')
        plt.imshow(img)

    plt.show()
