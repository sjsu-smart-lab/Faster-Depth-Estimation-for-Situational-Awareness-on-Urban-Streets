# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import time

import cv2
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import networks
import datasets
from layers import disp_to_depth
from utils import readlines
from utils import download_model_if_doesnt_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use')
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--quantization",
                        help='if set, enable quantization',
                        action='store_true')
    parser.add_argument("--object_level",
                        help='if set, enable object level',
                        action='store_true')
    parser.add_argument("--calibrate",
                        help='if set, enable quantization',
                        action='store_true')
    parser.add_argument('--train_data_path', type=str,
                        help='path to folder of train images for calibrating')

    return parser.parse_args()


def size_of_model(model):
    """ Print the size of the model.
    Args:
        model: model whose size needs to be determined
    """
    torch.save(model.state_dict(), "temp.p")
    # print('Size of the model(MB):', os.path.getsize("temp.p") / 1e6)
    model_size = os.path.getsize("temp.p") / 1e6
    os.remove('temp.p')
    return model_size


def get_num_param(model):
    params = sum([np.prod(p.size()) for p in model.parameters()])
    # print("Number of Parameters: %.1fM" % (params / 1e6))
    return params / 1e6


def print_total(encoder, decoder, name):
    tot = encoder + decoder
    print('{} = {:.2f}'.format(name, tot))


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("-> Loading model from ", args.model_name)
    encoder_path = os.path.join(args.model_name, "encoder.pth")
    depth_decoder_path = os.path.join(args.model_name, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = torch.load(encoder_path, map_location=device)
    if args.quantization:
        encoder.eval()
        encoder.fuse_model()
        encoder.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(encoder, inplace=True)
        if args.calibrate:
            assert args.train_data_path is not None, \
                "You must specify the --train_data_path parameter to calibrate; see README.md for an example"
            splits_dir = os.path.join(os.path.dirname(__file__), "splits")
            filenames = readlines(os.path.join(splits_dir, "eigen_zhou/train_files.txt"))
            dataset = datasets.KITTIRAWDataset(os.path.join(args.train_data_path, filenames),
                                               192, 640,
                                               [0], 4, is_train=False)
            dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=12,
                                    pin_memory=True, drop_last=False)
            encoder.eval()
            with torch.no_grad():
                print("[INFO] Calibrating...")
                for data in dataloader:
                    input_color = data[("color", 0, 0)]
                    input_color = input_color.to(device)
                    encoder(input_color)
            torch.quantization.convert(encoder, inplace=True)
            torch.save(encoder.state_dict(), os.path.join(args.load_weights_folder, 'qmodel.pth'))
        else:
            torch.quantization.convert(encoder, inplace=True)
            load_quant = torch.load(os.path.join(args.model_name, 'qmodel.pth'))
            encoder.load_state_dict(load_quant)

    encoder.to(device)
    # print(encoder)
    en_size = size_of_model(encoder)
    en_param = get_num_param(encoder)

    print("   Loading pretrained decoder")
    depth_decoder = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.to(device)
    depth_decoder.eval()
    de_size = size_of_model(depth_decoder)
    de_param = get_num_param(depth_decoder)

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    timings = list()
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            # in_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # print(in_image.shape)
            # original_width = in_image.shape[1]
            # original_height = in_image.shape[0]
            original_width, original_height = input_image.size
            input_image = input_image.resize((640, 192), pil.LANCZOS)
            # input_image = cv2.resize(in_image, (640, 192), interpolation=cv2.INTER_LANCZOS4)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            st = time.time()
            features = encoder(input_image)
            outputs = depth_decoder(features)
            et = time.time()
            print('Elapsed time = {:0.4f} ms'.format((et - st) * 1000))
            timings.append((et - st) * 1000)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()

            if args.object_level:
                for w in range(0, len(disp_resized_np), 16):
                    if w + 16 > len(disp_resized_np): break
                    for h in range(0, len(disp_resized_np[0]), 16):
                        if h + 16 > len(disp_resized_np[0]): break
                        disp_resized_np[w:w+16, h:h+16] = np.min(disp_resized_np[w:w+16, h:h+16])

            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            # overlay = cv2.cvtColor(np.array(colormapped_im), cv2.COLOR_RGB2BGR)
            # dst = cv2.addWeighted(overlay, 0.7, in_image, 1.0, 0.0)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)
            # cv2.imwrite(name_dest_im, dst)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(name_dest_npy))

        mean = np.mean(timings[11:])
        std = np.std(timings[11:])
        print('Mean time elapsed: {:0.4f}'.format(mean))
        print('Std Deviation: {:0.4f}'.format(std))

    print_total(en_size, de_size, 'Total model size')
    print_total(en_param, de_param, 'Total number of parameters')
    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
