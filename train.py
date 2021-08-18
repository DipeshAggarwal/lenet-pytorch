import matplotlib
matplotlib.use("Agg")

from core.lenet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="output/model.model", help="Path to output Trained Model")
ap.add_argument("-p", "--plot", type=str, default="output/plot.png", help="Path to Plot")
args = vars(ap.parse_args())

INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] Loading the KMNIST Dataset...")
train_data = KMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = KMNIST(root="data", train=False, download=True, transform=ToTensor())

print("[INFO] Generating the train/validation split...")
num_train_samples = int(len(train_data) * TRAIN_SPLIT)
num_val_samples = int(len(train_data) * VAL_SPLIT)
train_data, val_data = random_split(train_data, (num_train_samples, num_val_samples), generator=torch.Generator().manual_seed(42))

train_data_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

train_steps = len(train_data_loader.dataset) // BATCH_SIZE
val_steps = len(val_data_loader) // BATCH_SIZE

print("[INFO] Initialising the LeNet Model...")
model = LeNet(num_channels=1, classes=len(train_data.dataset.classes)).to(DEVICE)

opt = Adam(model.parameters(), lr=INIT_LR)
loss_func = nn.NLLLoss()

H = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

print("[INFO] Training the Network...")
start_time = time.time()

for e in range(0, EPOCHS):
    model.train()
    
    total_train_loss = 0
    total_val_loss = 0
    
    train_correct = 0
    val_correct = 0
    
    for x, y in train_data_loader:
        x, y = (x.to(DEVICE), y.to(DEVICE))
        
        pred = model(x)
        loss = loss_func(pred, y)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        total_train_loss += loss
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
    with torch.no_grad():
        model.eval()
        
        for x, y in val_data_loader:
            x, y = (x.to(DEVICE), y.to(DEVICE))
            
            pred = model(x)
            total_val_loss += loss_func(pred, y)
            val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    avg_train_loss = total_train_loss / train_steps
    avg_val_loss = total_val_loss / train_steps
    
    train_correct = train_correct / len(train_data_loader.dataset)
    val_correct = val_correct / len(val_data_loader.dataset)
    
    H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
    H["train_acc"].append(train_correct)
    H["val_loss"].append(avg_val_loss.cpu().detach().numpy())
    H["val_acc"].append(val_correct)
    
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avg_train_loss, train_correct))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avg_val_loss, val_correct))
    
end_time = time.time()
    
print("[INFO] total time taken to train the model: {:.2f}s".format(end_time - start_time)) 
print("[INFO] Evaluating Network")

with torch.no_grad():
    model.eval()
    
    preds = []
    
    for (x, y) in test_data_loader:
        x = x.to(DEVICE)
        pred = model(x)
        
        preds.extend(pred.argmax(axis=1).cpu().numpy())
        
print(classification_report(test_data.targets.cpu().numpy(), np.array(preds), target_names=test_data.classes))

plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

torch.save(model, args["model"])
