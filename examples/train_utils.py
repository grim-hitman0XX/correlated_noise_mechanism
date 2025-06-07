import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from opacus.utils.batch_memory_manager import BatchMemoryManager


def accuracy(preds, labels):
    return (preds == labels).mean()


def train(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    losses = []
    accuracy_list = []

    with BatchMemoryManager(
        data_loader=train_loader, max_physical_batch_size=128, optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            accuracy_list.append(acc)

            loss.backward()
            optimizer.step()

    print(
        f"\tTrain Epoch: {epoch} \t"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc@1: {np.mean(accuracy_list) * 100:.6f} "
    )
    return np.mean(accuracy_list) * 100


def train_no_dp(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    losses = []
    accuracy_list = []

    for i, (images, target) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()

        # measure accuracy and record loss
        acc = accuracy(preds, labels)

        losses.append(loss.item())
        accuracy_list.append(acc)

        loss.backward()
        optimizer.step()

    print(
        f"\tTrain Epoch: {epoch} \t"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc@1: {np.mean(accuracy_list) * 100:.6f} "
    )
    return np.mean(accuracy_list) * 100


def test(model, test_loader, criterion, device):
    model.eval()
    losses = []
    top1_acc = []
    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    avg_acc = np.mean(top1_acc)

    print(f"\tTest set:" f"Loss: {np.mean(losses):.6f} " f"Acc: {avg_acc * 100:.6f} ")
    return avg_acc
