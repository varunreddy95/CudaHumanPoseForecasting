from src.data.datasets import Hpose
from src.data.datasetsHeatmap import HposeHeat
from src.models.model2 import StateSpaceModel_Cell
from torch.utils.data import DataLoader
from src.configs.config import cfg
import torch
from torch import nn
from tqdm import tqdm
#import wandb
from src.util.visualization import draw_keypoint_sequence, draw_heatmap

from torch.utils.tensorboard import SummaryWriter
from paths import LOG_DIR, CHECKPOINT_DIR
import os

from src.util.visualization import draw_keypoint_sequence

def save_model(model, optimizer, epoch, logdir):
    """ Saving model checkpoint """

    if (not os.path.exists(os.path.join(CHECKPOINT_DIR, logdir))):
        os.mkdir(os.path.join(CHECKPOINT_DIR, logdir))
    savepath = os.path.join(CHECKPOINT_DIR, logdir, f"chekpoint_epoch_{epoch}.pth")
    #savepath = f"models/{name}_checkpoint_epoch_{epoch}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        #'stats': stats
    }, savepath)
    return


def load_model(model, optimizer, savepath):
    """ Loading pretrained checkpoint """

    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    #stats = checkpoint["stats"]

    return model, optimizer, epoch#, stats


def train_epoch(model, dataloader, optimizer, criterion, cfg):
    model.train()
    loss_sum = 0

    for batch, (x, labels) in enumerate(tqdm(dataloader)):
        x, labels = x.to(cfg.DEVICE), labels.to(cfg.DEVICE)

        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions, labels)

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

    # Compare for each 10 frames
    return loss_sum #/ (len(dataloader) * 10)


def train_epoch_tf(model, dataloader, optimizer, criterion, cfg):
    model.train()
    loss_sum = 0

    for batch, (x, labels) in enumerate(tqdm(dataloader)):
        x, labels = x.to(cfg.DEVICE), labels.to(cfg.DEVICE)

        optimizer.zero_grad()
        predictions = model(x, ground_truth=labels)
        loss = criterion(predictions, labels)

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

    # Compare for each 10 frames
    return loss_sum #/ (len(dataloader) * 10)

'''
# So far we don't need it 
def train_heatmap(model, dataloader, optimizer, criterion, cfg):
    model.train()
    loss_sum = 0

    for batch, (x, labels) in enumerate(tqdm(dataloader)):
        x, labels = x.to(cfg.DEVICE), labels.to(cfg.DEVICE)

        optimizer.zero_grad()
        predictions = model(x)
        combined_pred = torch.sum(predictions, 2)
        combined_labels = torch.sum(labels, 2)

        loss = criterion(combined_pred, combined_labels)

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

    return loss_sum
'''

def eval(model, dataloader, lossfunction, cfg):
    model.eval()
    correct = 0
    loss_sum = 0

    with torch.no_grad():
        for i, (x, labels) in enumerate(tqdm(dataloader)):
            x, labels = x.to(cfg.DEVICE), labels.to(cfg.DEVICE)

            predictions = model(x)
            loss = lossfunction(predictions, labels)
            loss_sum += loss.item()
            # _, predicted_labels = torch.max(predictions, 1)
            # predictions_tensor = torch.cat((predictions_tensor, predicted_labels))
            # ground_truth_tensor = torch.cat((ground_truth_tensor, labels))
            # # print(predicted_labels)
            # correct += torch.sum(predicted_labels == labels).item()

    # return correct, loss_sum / len(dataloader), predictions_tensor, ground_truth_tensor
    return loss_sum


def training(model, optimizer, scheduler, criterion, train_loader, valid_loader, logdir, cfg):
    """ Training a model for a given number of epochs"""
    if (not os.path.exists(os.path.join(LOG_DIR, logdir))):
        os.mkdir(os.path.join(LOG_DIR, logdir))
    writer = SummaryWriter(os.path.join(LOG_DIR, logdir))
    train_loss_list = []
    val_loss_list = []
    cfg.EPOCHS = 40

    eval_criterion = nn.MSELoss()

    best_val_loss = 10000.0
    epochs_better = 0

    for epoch in range(cfg.EPOCHS):

        # training epoch
        model.train()  # important for dropout and batch norms
        t_loss = train_epoch(
            model=model, dataloader=train_loader, optimizer=optimizer, criterion=criterion, cfg=cfg)
        scheduler.step()
        writer.add_scalar("Loss/train", t_loss, epoch)
        train_loss_list.append(t_loss)

        print("Epoch: {}/{}, Training loss: {:.3f}".format(epoch+1, cfg.EPOCHS, t_loss))
        #save_model(model, optimizer, epoch+1, "full")

        v_loss = eval(model, valid_loader, eval_criterion, cfg=cfg)
        val_loss_list.append(v_loss)

        writer.add_scalar("Loss/validation", v_loss, epoch)

        if v_loss < best_val_loss:
            save_model(model, optimizer, epoch+1, logdir)
            best_val_loss = v_loss
            epochs_better = 0
        else:
            epochs_better = epochs_better + 1

        
    seed_frames, gt = next(iter(valid_loader))
    model.eval()
    prediction = model(seed_frames.to(cfg.DEVICE))
    #seed_img, gt_img, pred_img = draw_keypoint_sequence(torch.mul(seed_frames, 1000.0), torch.mul(gt, 1000.0), torch.mul(prediction, 1000.0))
    seed_img, gt_img, pred_img = draw_heatmap(seed_frames, gt, prediction)

    writer.add_images("Seed Images", seed_img.detach().cpu(), 0)
    writer.add_images("Ground Truth", gt_img.detach().cpu(), 0)
    writer.add_images("Prediction", pred_img.detach().cpu(), 0)


    print(f"Training completed")
    return train_loss_list, val_loss_list

def training_joints(model, optimizer, scheduler, criterion, train_loader, valid_loader, logdir, cfg):
    if (not os.path.exists(os.path.join(LOG_DIR, logdir))):
        os.mkdir(os.path.join(LOG_DIR, logdir))
    """ Training a model for a given number of epochs"""
    writer = SummaryWriter(os.path.join(LOG_DIR, logdir))
    train_loss_list = []
    val_loss_list = []

    eval_criterion = nn.MSELoss()

    best_val_loss = 10000.0
    epochs_better = 0
    for epoch in range(cfg.EPOCHS):

        # training epoch
        model.train()  # important for dropout and batch norms
        t_loss = train_epoch(
            model=model, dataloader=train_loader, optimizer=optimizer, criterion=criterion, cfg=cfg)
        scheduler.step()
        writer.add_scalar("Loss/train", t_loss, epoch)
        train_loss_list.append(t_loss)

        print("Epoch: {}/{}, Training loss: {:.3f}".format(epoch+1, cfg.EPOCHS, t_loss))
        
        v_loss = eval(model, valid_loader, eval_criterion, cfg)
        val_loss_list.append(v_loss)

        if v_loss < best_val_loss:
            save_model(model, optimizer, epoch+1, logdir)
            best_val_loss = v_loss
            epochs_better = 0
        else:
            epochs_better = epochs_better + 1

        writer.add_scalar("Loss/validation", v_loss, epoch)

        # if epochs_better > 10:
        #     break

        if epoch%10 == 0:
            seed_frames, gt = next(iter(valid_loader))
            model.eval()
            prediction = model(seed_frames.to(cfg.DEVICE))
            seed_img, gt_img, pred_img = draw_keypoint_sequence(torch.mul(seed_frames, 1000.0), torch.mul(gt, 1000.0), torch.mul(prediction, 1000.0))
            #seed_img, gt_img, pred_img = draw_heatmap(seed_frames, gt, prediction)

            writer.add_images("Seed Images", seed_img.detach().cpu(), epoch)
            writer.add_images("Ground Truth", gt_img.detach().cpu(), epoch)
            writer.add_images("Prediction", pred_img.detach().cpu(), epoch)


    print(f"Training completed")
    return train_loss_list, val_loss_list

def training_tf(model, optimizer, scheduler, criterion, train_loader, valid_loader, logdir, cfg):
    if (not os.path.exists(os.path.join(LOG_DIR, logdir))):
        os.mkdir(os.path.join(LOG_DIR, logdir))
    """ Training a model for a given number of epochs"""
    writer = SummaryWriter(os.path.join(LOG_DIR, logdir))
    train_loss_list = []
    val_loss_list = []

    eval_criterion = nn.MSELoss()

    best_val_loss = 10000.0
    epochs_better = 0
    for epoch in range(cfg.EPOCHS):

        # training epoch
        model.train()  # important for dropout and batch norms
        t_loss = train_epoch_tf(
            model=model, dataloader=train_loader, optimizer=optimizer, criterion=criterion, cfg=cfg)
        scheduler.step()
        writer.add_scalar("Loss/train", t_loss, epoch)
        train_loss_list.append(t_loss)

        print("Epoch: {}/{}, Training loss: {:.3f}".format(epoch+1, cfg.EPOCHS, t_loss))
        
        v_loss = eval(model, valid_loader, eval_criterion, cfg)
        val_loss_list.append(v_loss)

        if v_loss < best_val_loss:
            save_model(model, optimizer, epoch+1, logdir)
            best_val_loss = v_loss
            epochs_better = 0
        else:
            epochs_better = epochs_better + 1

        writer.add_scalar("Loss/validation", v_loss, epoch)

        # if epochs_better > 10:
        #     break


        if epoch%10 == 0:
            seed_frames, gt = next(iter(valid_loader))
            model.eval()
            prediction = model(seed_frames.to(cfg.DEVICE))
            seed_img, gt_img, pred_img = draw_keypoint_sequence(torch.mul(seed_frames, 1000.0), torch.mul(gt, 1000.0), torch.mul(prediction, 1000.0))
            #seed_img, gt_img, pred_img = draw_heatmap(seed_frames, gt, prediction)

            writer.add_images("Seed Images", seed_img.detach().cpu(), epoch)
            writer.add_images("Ground Truth", gt_img.detach().cpu(), epoch)
            writer.add_images("Prediction", pred_img.detach().cpu(), epoch)

    print(f"Training completed")
    return train_loss_list, val_loss_list


def train_epoch_last_frames_first(model, dataloader, optimizer, criterion, cfg):
    model.train()
    loss_sum = 0

    for batch, (x, labels) in enumerate(tqdm(dataloader)):
        x, labels = x.to(cfg.DEVICE), labels.to(cfg.DEVICE)

        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions[:, 5:,:], labels[:, 5:,:])

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

    # Compare for each 10 frames
    return loss_sum #/ (len(dataloader) * 10)


def training_last_frames_first(model, optimizer, scheduler, criterion, train_loader, valid_loader, logdir, cfg):
    if (not os.path.exists(os.path.join(LOG_DIR, logdir))):
        os.mkdir(os.path.join(LOG_DIR, logdir))
    """ Training a model for a given number of epochs"""
    writer = SummaryWriter(os.path.join(LOG_DIR, logdir))
    train_loss_list = []
    val_loss_list = []

    eval_criterion = nn.MSELoss()

    best_val_loss = 10000.0
    epochs_better = 0
    for epoch in range(cfg.EPOCHS):

        # training epoch
        model.train()  # important for dropout and batch norms
        if epoch < 10:
            t_loss = train_epoch_last_frames_first(
                model=model, dataloader=train_loader, optimizer=optimizer, criterion=criterion, cfg=cfg)
        else: 
            t_loss = train_epoch(
                model=model, dataloader=train_loader, optimizer=optimizer, criterion=criterion, cfg=cfg)
        scheduler.step()
        writer.add_scalar("Loss/train", t_loss, epoch)
        train_loss_list.append(t_loss)

        print("Epoch: {}/{}, Training loss: {:.3f}".format(epoch+1, cfg.EPOCHS, t_loss))
        
        v_loss = eval(model, valid_loader, eval_criterion, cfg)
        val_loss_list.append(v_loss)

        if v_loss < best_val_loss:
            save_model(model, optimizer, epoch+1, logdir)
            best_val_loss = v_loss
            epochs_better = 0 
        else:
            epochs_better = epochs_better + 1

        writer.add_scalar("Loss/validation", v_loss, epoch)

        # if epochs_better > 10:
        #     break

        if epoch%10 == 0:
            seed_frames, gt = next(iter(valid_loader))
            model.eval()
            prediction = model(seed_frames.to(cfg.DEVICE))
            seed_img, gt_img, pred_img = draw_keypoint_sequence(torch.mul(seed_frames, 1000.0), torch.mul(gt, 1000.0), torch.mul(prediction, 1000.0))
            #seed_img, gt_img, pred_img = draw_heatmap(seed_frames, gt, prediction)

            writer.add_images("Seed Images", seed_img.detach().cpu(), epoch)
            writer.add_images("Ground Truth", gt_img.detach().cpu(), epoch)
            writer.add_images("Prediction", pred_img.detach().cpu(), epoch)


    print(f"Training completed")
    return train_loss_list, val_loss_list


def training_heatmap(model, optimizer, scheduler, criterion, train_loader, valid_loader, logdir, cfg):
    if (not os.path.exists(os.path.join(LOG_DIR, logdir))):
        os.mkdir(os.path.join(LOG_DIR, logdir))
    """ Training a model for a given number of epochs"""
    writer = SummaryWriter(os.path.join(LOG_DIR, logdir))
    train_loss_list = []
    val_loss_list = []

    eval_criterion = nn.MSELoss()

    best_val_loss = 10000.0
    epochs_better = 0
    for epoch in range(cfg.EPOCHS):

        # training epoch
        model.train()  # important for dropout and batch norms
        t_loss = train_epoch(
            model=model, dataloader=train_loader, optimizer=optimizer, criterion=criterion, cfg=cfg)
        scheduler.step()
        writer.add_scalar("Loss/train", t_loss, epoch)
        train_loss_list.append(t_loss)

        print("Epoch: {}/{}, Training loss: {:.3f}".format(epoch + 1, cfg.EPOCHS, t_loss))

        v_loss = eval(model, valid_loader, eval_criterion, cfg)
        val_loss_list.append(v_loss)

        save_model(model, optimizer, epoch + 1, logdir)
        #best_val_loss = v_loss

        writer.add_scalar("Loss/validation", v_loss, epoch)


    seed_frames, gt = next(iter(valid_loader))
    model.eval()
    prediction = model(seed_frames.to(cfg.DEVICE))
    seed_img, gt_img, pred_img = draw_heatmap(seed_frames, gt, prediction)

    writer.add_images("Seed Images", seed_img.detach().cpu(), 0)
    writer.add_images("Ground Truth", gt_img.detach().cpu(), 0)
    writer.add_images("Prediction", pred_img.detach().cpu(), 0)

    print(f"Training completed")
    return train_loss_list, val_loss_list

'''
def SecondModelMain():
    train_dataset = HposeHeat(subset="train")
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE)

#     valid_dataset = HposeHeat(subset="valid")
#     valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE)

    model = StateSpaceModel_Cell(encode_dim=128, hidden_dim=128).to(cfg.DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    training_loss, _ = training(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                                train_loader=train_loader, valid_loader=valid_loader)
'''

# if __name__ == '__main__':
#     SecondModelMain()
#     #firstModelMain()




