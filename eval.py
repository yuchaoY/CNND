import torch
import numpy as np

from utils import load_standard_mat, data_generator_mask, plot_roc_curve
from Dataset import Pixel
from model import CNND

import time
import os

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path setting
data_root = 'input/'
data_name = 'AVIRIS-II'
output_path = 'result/'
data_path = data_root + data_name + '.mat'
os.makedirs(output_path, exist_ok=True)

# Hyperparameters
inner_radius = 3
outer_radius = 5

# Data generator
hyperdata, gt = load_standard_mat(data_path, gt=True)
H, W = gt.shape
dataset = data_generator_mask(hyperdata, inner_radius, outer_radius)
data = Pixel(dataset)
gt = gt.reshape(-1)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=data,
                                           batch_size=1,
                                           shuffle=False)

# model definition
model = CNND().to(device)
model.load_state_dict(torch.load('model.ckpt', map_location=device))

start = time.time()
# eval model
model.eval()
anomal_detector = []
with torch.no_grad():

    for images in data_loader:
        images = images.to(device)
        images = images.permute(1, 0, 2)

        outputs = model(images)

        predicted = torch.mean(outputs).item()
        anomal_detector.append(predicted)

anomal_detector = np.array(anomal_detector)
end = time.time()
print('running time:', end - start)

plot_roc_curve(gt, anomal_detector, data_name)
anomal_detector = np.reshape(anomal_detector, (H, W))

np.savetxt('./result/'+data_name + '.csv', anomal_detector, delimiter=',')


