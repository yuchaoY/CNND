import torch
import numpy as np

from utils import load_standard_mat, data_generator_mask
from Dataset import Pixel
from model import CNND

import sklearn.metrics as metrics

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
inner_radius = 3
outer_radius = 5

# Data generator
data_root = 'input/'
data_name = 'AVIRIS-I.mat'
data_path = data_root + data_name
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
model.load_state_dict(torch.load('model.ckpt', map_location=torch.device('cpu')))

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

fpr, tpr, threshold = metrics.roc_curve(gt, anomal_detector)
roc_auc = metrics.auc(fpr, tpr)
print('roc_auc:', roc_auc)

anomal_detector = np.reshape(anomal_detector, (H, W))
np.savetxt('./result/'+data_name + '.csv', anomal_detector, delimiter=',', fmt='%.2f')


