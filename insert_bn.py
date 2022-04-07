import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
import torch.utils.data.distributed
from utils import readlines, sec_to_hm_str
# from torchvision import transforms, datasets
import datasets
from trainer import Trainer
from options import MonodepthOptions
# from utils import accuracy, ProgressMeter, AverageMeter
from networks import RepVGGBlock
# from utils import load_checkpoint, get_ImageNet_train_dataset, get_default_train_trans

#   Insert BN into an inference-time RepVGG (e.g., for quantization-aware training).
#   Get the mean and std on every conv3x3 (before the bias-adding) on the train set. Then use such data to initialize
#   BN layers and insert them after conv3x3.
#   May, 07, 2021


def update_running_mean_var(x, running_mean, running_var, momentum=0.9, is_first_batch=False):
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    if is_first_batch:
        running_mean = mean
        running_var = var
    else:
        running_mean = momentum * running_mean + (1.0 - momentum) * mean
        running_var = momentum * running_var + (1.0 - momentum) * var
    return running_mean, running_var


#   Record the mean and std like a BN layer but do no normalization
class BNStatistics(nn.Module):
    def __init__(self, num_features):
        super(BNStatistics, self).__init__()
        shape = (1, num_features, 1, 1)
        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.zeros(shape))
        self.is_first_batch = True

    def forward(self, x):
        if self.running_mean.device != x.device:
            self.running_mean = self.running_mean.to(x.device)
            self.running_var = self.running_var.to(x.device)
        self.running_mean, self.running_var = update_running_mean_var(x, self.running_mean, self.running_var,
                                                                      momentum=0.9, is_first_batch=self.is_first_batch)
        self.is_first_batch = False
        return x


#   This is designed to insert BNStat layer between Conv2d(without bias) and its bias
class BiasAdd(nn.Module):
    def __init__(self, num_features):
        super(BiasAdd, self).__init__()
        self.bias = torch.nn.Parameter(torch.Tensor(num_features))

    def forward(self, x):
        return x + self.bias.view(1, -1, 1, 1)


def switch_repvggblock_to_bnstat(model):
    for n, block in model.named_modules():
        if isinstance(block, RepVGGBlock):
            print('switch to BN Statistics: ', n)
            assert hasattr(block, 'rbr_reparam')
            stat = nn.Sequential()
            stat.add_module('conv', nn.Conv2d(block.rbr_reparam.in_channels, block.rbr_reparam.out_channels,
                                              block.rbr_reparam.kernel_size,
                                              block.rbr_reparam.stride, block.rbr_reparam.padding,
                                              block.rbr_reparam.dilation,
                                              block.rbr_reparam.groups, bias=False))  # Note bias=False
            stat.add_module('bnstat', BNStatistics(block.rbr_reparam.out_channels))
            stat.add_module('biasadd', BiasAdd(block.rbr_reparam.out_channels))  # Bias is here
            stat.conv.weight.data = block.rbr_reparam.weight.data
            stat.biasadd.bias.data = block.rbr_reparam.bias.data
            block.__delattr__('rbr_reparam')
            block.rbr_reparam = stat


def switch_bnstat_to_convbn(model):
    for n, block in model.named_modules():
        if isinstance(block, RepVGGBlock):
            assert hasattr(block, 'rbr_reparam')
            assert hasattr(block.rbr_reparam, 'bnstat')
            print('switch to ConvBN: ', n)
            conv = nn.Conv2d(block.rbr_reparam.conv.in_channels, block.rbr_reparam.conv.out_channels,
                             block.rbr_reparam.conv.kernel_size,
                             block.rbr_reparam.conv.stride, block.rbr_reparam.conv.padding,
                             block.rbr_reparam.conv.dilation,
                             block.rbr_reparam.conv.groups, bias=False)
            bn = nn.BatchNorm2d(block.rbr_reparam.conv.out_channels)
            # Initialize the mean and var of BN with the statistics
            bn.running_mean = block.rbr_reparam.bnstat.running_mean.squeeze()
            bn.running_var = block.rbr_reparam.bnstat.running_var.squeeze()
            std = (bn.running_var + bn.eps).sqrt()
            conv.weight.data = block.rbr_reparam.conv.weight.data
            bn.weight.data = std
            # Initialize gamma = std and beta = bias + mean
            bn.bias.data = block.rbr_reparam.biasadd.bias.data + bn.running_mean

            convbn = nn.Sequential()
            convbn.add_module('conv', conv)
            convbn.add_module('bn', bn)
            block.__delattr__('rbr_reparam')
            block.rbr_reparam = convbn


#   Insert a BN after conv3x3 (rbr_reparam). With no reasonable initialization of BN, the model may break down.
#   So you have to load the weights obtained through the BN statistics
#   (please see the function "insert_bn" in this file).
def directly_insert_bn_without_init(model):
    for n, block in model.named_modules():
        if isinstance(block, RepVGGBlock):
            print('directly insert a BN with no initialization: ', n)
            assert hasattr(block, 'rbr_reparam')
            convbn = nn.Sequential()
            convbn.add_module('conv', nn.Conv2d(block.rbr_reparam.in_channels, block.rbr_reparam.out_channels,
                                                block.rbr_reparam.kernel_size,
                                                block.rbr_reparam.stride, block.rbr_reparam.padding,
                                                block.rbr_reparam.dilation,
                                                block.rbr_reparam.groups, bias=False))  # Note bias=False
            convbn.add_module('bn', nn.BatchNorm2d(block.rbr_reparam.out_channels))
            #   ====================
            convbn.add_module('relu', nn.ReLU())
            # TODO we moved ReLU from "block.nonlinearity" into "rbr_reparam" (nn.Sequential).
            #  This makes it more convenient to fuse operators (see RepVGGWholeQuant.fuse_model)
            #  using off-the-shelf APIs.
            block.nonlinearity = nn.Identity()
            # ==========================
            block.__delattr__('rbr_reparam')
            block.rbr_reparam = convbn


def log_time(batch_idx, duration, loss, args):
    """Print a logging statement to the terminal
    """
    samples_per_sec = args.batch_size / duration
    print_string = "batch {:>6} | examples/s: {:5.1f}" + \
                   " | loss: {:.5f}"
    print(print_string.format(batch_idx, samples_per_sec, loss))


def insert_bn(args, trainer):
    # args = parser.parse_args()
    device = 'cuda'

    print("loading model from folder {}".format(args.load_weights_folder))
    encoder_path = os.path.join(args.load_weights_folder, "encoder_noBN.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")
    encoder = torch.load(encoder_path, map_location=device)
    switch_repvggblock_to_bnstat(encoder)
    decoder = torch.load(decoder_path, map_location=device)

    # cudnn.benchmark = True
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")
    filenames = readlines(os.path.join(splits_dir, args.split, "train_files.txt"))
    img_ext = '.png' if args.png else '.jpg'
    dataset = datasets.KITTIRAWDataset(args.data_path, filenames,
                                       192, 640, args.frame_ids, 4,
                                       is_train=True, img_ext=img_ext)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)
    step = 0
    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            end = time.time()
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(device)

            features = encoder(inputs["color_aug", 0, 0])
            outputs = decoder(features)
            if trainer.use_pose_net:
                outputs.update(trainer.predict_poses(inputs, features))
            trainer.generate_images_pred(inputs, outputs)
            losses = trainer.compute_losses(inputs, outputs)

            # measure elapsed time
            duration = time.time() - end

            early_phase = batch_idx % args.log_frequency == 0 and step < 2000
            late_phase = step % 2000 == 0

            if early_phase or late_phase:
                log_time(batch_idx, duration, losses["loss"].cpu().data, args)

                if "depth_gt" in inputs:
                    trainer.compute_depth_losses(inputs, outputs, losses)
            step += 1

    switch_bnstat_to_convbn(encoder)
    encoderbn_path = os.path.join(args.load_weights_folder, "encoder.pth")
    torch.save(encoder, encoderbn_path)


if __name__ == '__main__':
    options = MonodepthOptions()
    trainer = Trainer(options.parse())
    insert_bn(options.parse(), trainer)
