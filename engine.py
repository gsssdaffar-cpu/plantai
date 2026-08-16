import torch
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0
    correct = 0
    total = 0

    progress = tqdm(loader)

    for images, labels in progress:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        progress.set_description(
            f"Loss {loss.item():.4f}"
        )

    accuracy = 100 * correct / total

    return running_loss / len(loader), accuracy


def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return running_loss / len(loader), accuracy