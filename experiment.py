import time
import os
import networks
import numpy as np
import argparse
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10, MNIST, SVHN
from torchvision import transforms
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--model', type=str, required=True, choices=['resnet', 'vgg', 'inception', 'mobilenetv2',
                                                                 'mobilenetv3', 'resnet50', 'efficientnet'])
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--name_to_save', type=str, default='model')
parser.add_argument('--load_weights', type=str)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--quantize', action='store_true', default=False)
parser.add_argument('--agg_prune', action='store_true', default=False)
parser.add_argument('--dataset', type=str, required=True, choices=['cifar', 'mnist', 'svhn'])

args = parser.parse_args()

if args.dataset == 'cifar':
    mnist = False
elif args.dataset == 'mnist':
    mnist = True
elif args.dataset == 'svhn':
    mnist = False

if args.model == 'resnet':
    # model_to_train = resnet18.ResNet
    # modules_to_fuse = resnet18.modules_to_fuse_resnet()
    # prune_model = resnet18.prune_model_resnet
    pass
elif args.model == 'vgg':
    # model_to_train = VGG19.vgg19_bn
    # modules_to_fuse = VGG19.modules_to_fuse_vgg()
    # prune_model = VGG19.prune_model_vgg
    pass
elif args.model == 'inception':
    # model_to_train = inception.inception_v3
    # modules_to_fuse = inception.modules_to_fuse_inception()
    # prune_model = inception.prune_model_inception
    pass
elif args.model == 'mobilenetv2':
    model_to_train = networks.ExpModel(mnist=mnist)

elif args.model == 'mobilenetv3':
    model_to_train = networks.ExpModel_Mobv3(mnist=mnist)

elif args.model == 'efficientnet':
    model_to_train = networks.ExpModel_Effnet(mnist=mnist)

elif args.model == 'resnet50':
    # model_to_train = resnet50.resnet50
    # modules_to_fuse = resnet50.modules_to_fuse_resnet50()
    # prune_model = resnet50.prune_model_resnet50
    pass


def get_dataloader():
    if args.dataset == 'svhn':
        train_loader = torch.utils.data.DataLoader(
            SVHN('./data', split='train', transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]), download=True), batch_size=args.batch_size, num_workers=1,
            shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            SVHN('./data', split='test', transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]), download=True), batch_size=args.batch_size, num_workers=1,
            shuffle=True, pin_memory=True)
    elif args.dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            MNIST('./data', train=True, transform=transforms.Compose([
                transforms.Resize([32, 32]),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]), download=True), batch_size=args.batch_size, num_workers=1,
            shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            MNIST('./data', train=False, transform=transforms.Compose([
                transforms.Resize([32, 32]),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]), download=True), batch_size=args.batch_size, num_workers=1,
            shuffle=True, pin_memory=True)
    elif args.dataset == 'cifar':
        train_loader = torch.utils.data.DataLoader(
            CIFAR10('./data', train=True, transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]), download=True), batch_size=args.batch_size, num_workers=1,
            shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]), download=True), batch_size=args.batch_size, num_workers=1,
            shuffle=True, pin_memory=True)
    return train_loader, test_loader


def print_size_of_model(model):
    """ Print the size of the model.
    Args:
        model: model whose size needs to be determined
    """
    torch.save(model.state_dict(), "temp.p")
    print('Size of the model(MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def calibrate_model(model, loader):
    print("Calibrating...")
    device = args.device
    model.to(device)
    model.eval()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        _ = model(inputs)


def eval(model, test_loader, quantize=False, train_loader=None):
    correct = 0
    total = 0
    timings = list()
    device = args.device
    model.to(device)
    model.eval()
    if quantize:
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        calibrate_model(model, train_loader)
        torch.quantization.convert(model, inplace=True)
        model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            st = time.time()
            out = model(img)
            et = time.time()
            timings.append((et - st) * 1000)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred == target).sum()
            total += len(target)
    print_size_of_model(model)
    return correct / total, timings


def train_model(model, train_loader, test_loader):
    device = args.device
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)
    model.to(device)

    best_acc = -1
    for epoch in range(args.total_epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
        model.eval()
        acc, _ = eval(model, test_loader)
        print("Epoch %d/%d, Acc=%.4f, Loss=%.4f" % (epoch, args.total_epochs, acc, loss.item()))
        if best_acc < acc:
            torch.save(model, args.name_to_save)
            print("===Model at epoch %d saved===" % epoch)
            best_acc = acc
        scheduler.step()
    print("Best Acc=%.4f" % best_acc)


def main():
    train_loader, test_loader = get_dataloader()
    if args.mode == 'train':
        args.round = 0
        model = model_to_train
        train_model(model, train_loader, test_loader)
    elif args.mode == 'prune':
        previous_ckpt = args.load_weights
        print("Pruning model from %s" % previous_ckpt)
        model = torch.load(previous_ckpt)
        model,  _, _ = model.prune(model, agg=args.agg_prune)
        print(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM" % (params / 1e6))
        train_model(model, train_loader, test_loader)
    elif args.mode == 'test':
        ckpt = args.load_weights
        print("Load model from %s" % ckpt)
        model = torch.load(ckpt)
        acc, timings = eval(model, test_loader, quantize=args.quantize, train_loader=train_loader)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        mean = np.mean(timings[11:])
        std = np.std(timings[11:])
        print("Number of Parameters: %.1fM" % (params / 1e6))
        print('Mean time elapsed: {:0.4f}'.format(mean))
        print('Std time elapsed: {:0.4f}'.format(std))
        print("Acc=%.4f\n" % acc)
        return mean, std


if __name__ == '__main__':
    main()
    # avg_mean = 0
    # avg_std = 0
    # for i in range(3):
    #     mean, std = main()
    #     avg_mean += mean
    #     avg_std += std
    # print('avg_mean = {:0.2f}; avg_std = {:0.2f}'. format(avg_mean/3, avg_std/3))
