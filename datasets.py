import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# converts black and white images to color by stacking
class Grey2Color:
    def __call__(self, img):
        img = torch.vstack((img, img, img)) 
        return img

# constant random data
class FakeData(Dataset):
    def __init__(self, dataset_type):
        assert dataset_type in ['constant', 'random']
        self.dataset_type = dataset_type
        self.len = int(1e5)

    def __len__(self):
        return self.len

    def __getitem__(self, _):
        if self.dataset_type == 'constant':
            f = torch.rand(3, 1, 1)
            x = f * torch.ones(3, 32, 32)
        elif self.dataset_type == 'random':
            x = torch.rand((3, 32, 32))
        else: raise NotImplementedError

        return x, torch.zeros(1)

# list of all datasets 
def all_datasets():
    datasets = [
        'constant', 'random',
        'mnist', 'fmnist', 'kmnist', 'cifar10',
        'svhn', 'celeba',
        'lsun',
        'cifar100', 'imagenet'#, 'cifar10',
    ]
    return datasets

def get_loader(path, dataset_name, split, batch_size):
    assert split in ['train', 'test']
    assert dataset_name in all_datasets() 

    train = split == 'train'

    # resize everything to 32x32
    colour_transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
    ])

    # if grey, convert to color by stacking channels
    grey_transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        Grey2Color(),
    ]) 

    # get dataset, assumes that '__main__' was run to download images already
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(root=path, train=train, download=False, transform=grey_transforms)
    elif dataset_name == 'fmnist':
        dataset = datasets.FashionMNIST(root=path, train=train, download=False, transform=grey_transforms)
    elif dataset_name == 'kmnist':
        dataset = datasets.KMNIST(root=path, train=train, download=False, transform=grey_transforms)
    elif dataset_name == 'svhn':
        dataset = datasets.SVHN(root=path, split=split, download=False, transform=colour_transforms)
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root=path, train=train, download=False, transform=colour_transforms)
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root=path, train=train, download=False, transform=colour_transforms)
    elif dataset_name == 'celeba':
        dataset = datasets.CelebA(root=path, split=split, download=False, transform=colour_transforms)
    elif dataset_name == 'lsun':
        split = 'bedroom_train' #if train else 'bedroom_val'
        path = '/drive2/ood/lsun/'
        dataset = datasets.LSUN(root=path, classes=[split], transform=colour_transforms)
    elif dataset_name == 'imagenet':
        split = 'train'
        dataset = datasets.ImageNet(root=path, split=split, transform=colour_transforms)
    elif dataset_name in ['constant', 'random']:
        dataset = FakeData(dataset_type=dataset_name)
    else: raise ValueError(f'invalid dataset name: {dataset_name}')

    # get loader
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return loader

def get_latents(path, dataset, mode):
    assert mode in ['train', 'test']

    if mode == 'train':
        z = torch.load(os.path.join(path, 'autoenc_lat', f'{dataset}_train_z.pt'))
    else:
        z = torch.load(os.path.join(path, 'tuned_lat', f'{dataset}_z.pt'))
    return z

# convenience function for getting latent loader
def get_lat_loader(save_path, dataset, mode, batch_size=64):
    assert mode in ['train', 'test']
    if mode == 'test': batch_size = 32

    z = get_latents(save_path, dataset, mode) 
    loader = LatentLoader(z, batch_size=batch_size, shuffle=True)
    if mode == 'test':
        z_loss = torch.load(os.path.join(save_path, 'tuned_lat', f'{dataset}_loss.pt'))
        return loader, z_loss
    else:
        return loader

class LatentLoader:
    def __init__(self, x, batch_size, shuffle=True):
        self.x = x
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_batch = int(np.ceil(self.x.shape[0] / self.batch_size))

        self.i = 0
        if self.shuffle:
            self.ind = np.random.permutation(np.arange(x.shape[0]))
        else:
            self.ind = np.arange(x.shape[0])

        self.dim = x.shape[1]
    
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.n_batch - 1:
            ind = self.ind[self.i * self.batch_size : (self.i + 1) * self.batch_size]
        elif self.i == self.n_batch - 1:
            ind = self.ind[self.i * self.batch_size : ]
        else:
            self.i = 0
            if self.shuffle: self.shuffle_ind()
            raise StopIteration

        x_ = self.x[ind]
        self.i += 1

        return x_ 

    def shuffle_ind(self):
        self.ind = np.random.permutation(self.ind)

if __name__ == '__main__':
    path = '/drive2/ood/'

    names = ['mnist', 'fmnist', 'kmnist', 'svhn', 'cifar10', 'cifar100', 'celeba', 'constant', 'random', 'imagenet']
    for name in names:
        # plot some images
        loader = get_loader(path, name, 'train', 2, 1)
        for x, _ in loader:
            print(x.shape)
            plt.imshow(x[0].permute(1, 2, 0))
            plt.title(name)
            plt.show()
            break
