import torch

# for training data
class DifferPixelPair(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.transform = transform
        self.data, self.labels = data, labels

    def __getitem__(self, item):
        img, target = self.data[item], self.labels[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

# for testing data
class Pixel(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __getitem__(self, item):
        img = self.data[item]
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)
