import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

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
        'mnist', 'fmnist', 'kmnist', 'svhn', 'cifar10', 'cifar100', 
        'celeba', 'constant', 'random',
    ]
    return datasets

def get_loader(path, dataset_name, split, batch_size, workers):
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
        dataset = datasets.MNIST(root=path, train=train, download=True, transform=grey_transforms)
    elif dataset_name == 'fmnist':
        dataset = datasets.FashionMNIST(root=path, train=train, download=True, transform=grey_transforms)
    elif dataset_name == 'kmnist':
        dataset = datasets.KMNIST(root=path, train=train, download=True, transform=grey_transforms)
    elif dataset_name == 'svhn':
        dataset = datasets.SVHN(root=path, split=split, download=True, transform=colour_transforms)
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root=path, train=train, download=True, transform=colour_transforms)
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root=path, train=train, download=True, transform=colour_transforms)
    elif dataset_name == 'celeba':
        dataset = datasets.CelebA(root=path, split=split, download=True, transform=colour_transforms)
    elif dataset_name == 'imagenet':
        dataset = datasets.ImageNet(root=path, split=split, transform=colour_transforms)
    elif dataset_name in ['constant', 'random']:
        dataset = FakeData(dataset_type=dataset_name)
    else: raise NotImplementedError

    # get loader
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    return loader

'''
download datasets
celeba and imagenet require external downloads

celeba:
    - celeba requires: TODO: list files in path/celeba/

imagenet:
    - imagenet requires: wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
'''

import matplotlib.pyplot as plt

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

