import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from Dataset import DifferPixelPair
from model import CNND
from utils import training_data_generator

writer = SummaryWriter('./experiments/real')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
learning_rate = 0.001
batch_size = 256
num_epochs = 50

# Data generator
data_path = 'input/Salinas.mat'
gt_path = 'input/Salinas_gt.mat'
data, labels = training_data_generator(data_path, gt_path, sim_samples=100,
                                       dis_samples=600, remove_bands=True) # data:(N, 1, d), labels:(N, )
salinas = DifferPixelPair(data, labels)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=salinas,
                                           batch_size=batch_size,
                                           shuffle=True)

# Model definition
model = CNND().to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            writer.add_scalar('Loss/loss', loss.item(), i+epoch*total_step)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

torch.save(model.state_dict(), 'model.ckpt')