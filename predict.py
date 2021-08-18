import numpy as np
np.random.seed(42)

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import argparse
import imutils
import torch
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="output/model.model", help="Path to the Model")
args = vars(ap.parse_args())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] Loading the KMNIST test dataset...")
test_data = KMNIST(root="data", train=False, download=True, transform=ToTensor())
idxs = np.random.choice(range(0, len(test_data)), size=(10,))
test_data = Subset(test_data, idxs)

test_data_loader = DataLoader(test_data, batch_size=1)

model = torch.load(args["model"]).to(DEVICE)
model.eval()

with torch.no_grad():
    for image, label in test_data_loader:
        orig_image = image.numpy().squeeze(axis=(0, 1))
        gt_label = test_data.dataset.classes[label.numpy()[0]]
        
        image = image.to(DEVICE)
        pred = model(image)
        
        idx = pred.argmax(axis=1).cpu().numpy()[0]
        pred_label = test_data.dataset.classes[idx]
        
        orig_image = np.dstack([orig_image] * 3)
        orig_image = imutils.resize(orig_image, width=128)
        
        color = (0, 255, 0) if gt_label == pred_label else (255, 0, 0)
        cv2.putText(orig_image, gt_label, (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
        
        print("[INFO] Ground Truth label: {}, predicted label: {}".format(gt_label, pred_label))
        cv2.imshow("image", orig_image)
        cv2.waitKey(0)
        