import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

def train(model, optimizer, dataloader, device, criterion = F.nll_loss):
    avg_loss, correct = 0, 0
    model.train()
    for data, target in tqdm.tqdm(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        pred = output.argmax(dim = 1, keepdim = True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss /= len(dataloader.dataset)
    accuracy = 100. * correct / len(dataloader.dataset)
    
    print(f"Train set: Average Loss: {avg_loss:.6f}\tAccuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy

def validate(model, dataloader, device, criterion = F.nll_loss):
    correct = 0
    avg_loss = 0
    all_preds, all_targets = [], []

    model.eval()
    with torch.no_grad():
        for data, target in tqdm.tqdm(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            avg_loss = criterion(output, target).item()

            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss /= len(dataloader.dataset)
    
    f1 = f1_score(all_preds, all_targets, average='weighted')
    accuracy = 100. * correct / len(dataloader.dataset)

    print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%), F1-Score {:.2f}\n'.format(
        avg_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset),
        f1))

    return avg_loss, accuracy, f1