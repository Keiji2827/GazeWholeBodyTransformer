

import argparse
import os

import torch
from hrnet.config import config as hrnet_config
from hrnet.config import update_config as hrnet_update_config
from hrnet.hrnet_cls_net_featmaps import get_cls_net
from utils.logger import setup_logger
from utils.miscellaneous import mkdir

#from dataloader.gafa_loader import create_gafa_dataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')

    #########################################################
    # Others
    #########################################################
    parser.add_argument("--device", type=str, default='cuda',
                        help="cuda or cpu")

    args = parser.parse_args()
    return args

def main(args):

    print("in main")
    global logger
    mkdir(args.output_dir)
    logger = setup_logger("model Test", args.output_dir, 0)


    args.device = torch.device(args.device)

    # init ImageNet pre-trained backbone model
    if args.arch=='hrnet':
        hrnet_yml = 'models/hrnet/weights/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_checkpoint = './models/hrnet/weights/hrnetv2_w40_imagenet_pretrained.pth'
        hrnet_update_config(hrnet_config, hrnet_yml)
        backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
        logger.info('=> loading hrnet-v2-w40 model')

    elif args.arch=='hrnet-w64':
        hrnet_yaml = 'models/hrnet/weights/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_checkpoint = 'models/hrnet/weights/hrnetv2_w64_imagenet_pretrained.pth'
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
        logger.info('=> loading hrnet-v2-w64 model')
    else:
        print("=> using pre-trained model '{}'".format(args.arch))
        backbone = models.__dict__[args.arch](pretrained=True)
        # remove the last fc layer
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

    backbone



    return



if __name__ == "__main__":
    args = parse_args()
    main(args)
