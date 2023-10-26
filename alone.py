from model.vgg import vgg19
import argparse
import os
from dataset.dataset import Crowd2
from torch.utils.data import DataLoader
import torch
import numpy as np


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='inach',
                        help='The path to data directory')
    parser.add_argument('--saved-model', default='history/pretrained/best_model.pth',
                        help='model directory')
    parser.add_argument('--device', default='0',
                        help='assign device')
    parser.add_argument('--pretrained', default=False,
                        help='the path to the pretrained model')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()

    datasets = Crowd2(args.data_dir, 448, 8, 'val')
    dataloader = DataLoader(datasets, 1, shuffle=False, pin_memory=False)

    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(args.saved_model, device))

    model.eval()
    for inputs,  name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1
        with torch.set_grad_enabled(False):
            den, bg = model(inputs)
            pred = (((den * (bg >= 0.5)))).sum().item()

            print(name[0],  pred )





