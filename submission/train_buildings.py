# Authors: Arnav Jain, Solomon Starkes, and Sidy Thiam

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision
from model import CNN

from dataset import BuildingDataset

import numpy as np

import os, sys

IMAGE_HEIGHT = 189
IMAGE_WIDTH = 252

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def split_train_val(org_train_set, valid_ratio=0.1):

    num_train = len(org_train_set)

    split = int(np.floor(valid_ratio * num_train))        

    indices = list(range(num_train))

    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    new_train_set = Subset(org_train_set, train_idx)
    val_set = Subset(org_train_set, val_idx)

    assert num_train - split == len(new_train_set)
    assert split == len(val_set)

    return new_train_set, val_set

def test(net, loader, device):
    # prepare model for testing (only important for dropout, batch norm, etc.)
    net.eval()
    
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:

            data, target = data.to(device), target.to(device)
            
            output = net(data)
            log_probs = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(log_probs, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += (pred.eq(target.data.view_as(pred)).sum().item())
            
            total = total + 1

    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(loader.dataset),
        (100. * correct / len(loader.dataset))), flush=True)
    
    return 100.0 * correct / len(loader.dataset)

def train(net, loader, optimizer, epoch, device, log_interval=1):
    # prepare model for training (only important for dropout, batch norm, etc.)
    net.train()

    correct = 0
    for batch_idx, (data, target) in enumerate(loader):

        data, target = data.to(device), target.to(device)
        
        # clear up gradients for backprop
        optimizer.zero_grad()
        output = F.log_softmax(net(data), dim=1)

        # use NLL loss
        loss = F.nll_loss(output, target)

        # compute gradients and make updates
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct += (pred.eq(target.data.view_as(pred)).sum().item())

        if batch_idx % log_interval == 0:
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset), 100. * batch_idx / len(loader), loss.item()), flush=True)

    print('\tAccuracy: {:.2f}%'.format(100.0 * correct / len(loader.dataset)), flush=True)  


if __name__ == '__main__':

    # image parameters
    resize_factor = 1
    new_h = int(IMAGE_HEIGHT / resize_factor)
    new_w = int(IMAGE_WIDTH / resize_factor)

    # feel free to change the hardcoded numbers -- they are from the CIFAR10 code
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    resize = torchvision.transforms.Resize(size = (new_h, new_w))
    convert = torchvision.transforms.ConvertImageDtype(torch.float)
    
    train_transforms = torchvision.transforms.Compose([resize, convert, normalize])
    test_transforms = torchvision.transforms.Compose([resize, convert, normalize])

    data_folder = '/gpfs/u/scratch/RNL3/shared/all_images'

    train_dataset = BuildingDataset('train_labels.csv', data_folder, transform=train_transforms)
    test_dataset = BuildingDataset('val_labels.csv', data_folder, transform=test_transforms)


    # Plotting (leaving this here in case you'd like to take a look)
    # image = train_dataset[10][0]
    # image = image.permute(1,2,0)

    # plt.figure()
    # plt.imshow(image)
    # #plt.imshow(torch.reshape(image, (new_h, new_w)), cmap='gray_r')
    # plt.show()

    # set training hyperparameters
    train_batch_size = 100
    test_batch_size = 100
    n_epochs = 30
    learning_rate = 1e-3
    seed = 100
    input_dim = (3, new_h, new_w)
    out_dim = 11
    momentum = 0.9

    # put data into loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    network = CNN(in_dim=input_dim, out_dim=out_dim)
    network = network.to(device)

    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        
    PATH = model_path + 'cnn_buildings.pth'
    
    if len(sys.argv) > 1 and sys.argv[1] == 'load':
        network.load_state_dict(torch.load(PATH))

    # sanity check -- output should be close to 1/11
    print('Initial accuracy', flush=True)
    test(network, test_loader, device)

    # training loop
    best_acc = 0.0
    best_epoch = 0
    best_state_dict = None
    for epoch in range(1, n_epochs + 1):
        train(network, train_loader, optimizer, epoch, device)
        acc = test(network, test_loader, device)
        if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_state_dict = network.state_dict()		

    if best_state_dict is not None:
        torch.save(best_state_dict, PATH)
        print(f"Saved best model at epoch {best_epoch} with accuracy = {best_acc:.2f}%")
    else:
        torch.save(network.state_dict(), PATH)
