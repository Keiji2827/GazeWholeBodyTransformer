

import argparse
import os
import os.path as op
import torch
from models.bert.modeling_bert import BertConfig
from models.bert.modeling_metro import METRO
from models.bert.model_bert import GAZEBERT_Network as GAZEBERT
from models.hrnet.config import config as hrnet_config
from models.hrnet.config import update_config as hrnet_update_config
from models.hrnet.hrnet_cls_net_featmaps import get_cls_net
from models.utils.logger import setup_logger
from models.utils.miscellaneous import mkdir

from PIL import Image
from torchvision import transforms

#from dataloader.gafa_loader import create_gafa_dataset

transform = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

transform_visualize = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])


def run_inference(args, image_list, _gaze_bert):

    _gaze_bert.eval()

    for image_file in image_list:
        if "pred" not in image_file:
            print(image_file)

            img = Image.open(image_file)
            # from torchvision import transforms
            img_tensor = transform(img)
            #img_visual = transform_visualize(img)

            batch_imgs = torch.unsqueeze(img_tensor, 0)#.cuda()
            #batch_visual_imgs = torch.unsqueeze(img_visual, 0).cuda()

            output = _gaze_bert(batch_imgs)
            #output = backbone(batch_imgs)
            #print(image_file)
            #print(output)
            print(output.size())




    return



def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--image_file_or_path", default='./data/sample', type=str, 
                        help="test data")
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='models/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--drop_out", default=0.1, type=float, 
                        help="Drop out ratio in BERT.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                        help="The Image Feature Dimension.")          
    parser.add_argument("--hidden_feat_dim", default='1024,256,128', type=str, 
                        help="The Image Feature Dimension.")   
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--run_eval_only", default=False, action='store_true',) 
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

    # Load model
    trans_encoder = []

    input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
    output_feat_dim = input_feat_dim[1:]+[3]

    if args.run_eval_only==True : 
        # if only run eval, load checkpoint
        # not use at 22-11-30
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _gaze_bert = torch.load(args.resume_checkpoint)
    else:
        # init three transformer-encoder blocks in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, METRO
            config = config_class.from_pretrained(args.config_name if args.config_name else \
                                                    args.model_name_or_path)
            print(type(config))
            config.output_attentions = False
            config.hidden_dropout_prob = args.drop_out
            config.img_feature_dim = input_feat_dim[i]
            config.output_feature_dim = hidden_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]
            args.intermediate_size = -1

            # update model structure if specified in argments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
                config_param = getattr(args, param)
                if arg_param > 0 and arg_param != config_param:
                    logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)
        
            # init ImageNet pre-trained backbone model
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config)
            logger.info("Init model from scratch.")
            trans_encoder.append(model)
                
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

    backbone_total_param = sum(p.numel() for p in backbone.parameters())
    logger.info("Backbone total parameters: {}".format(backbone_total_param))


    # Initialize GAZEBERT model 
    _gaze_bert = GAZEBERT(args, backbone)



    image_list = []

    if not args.image_file_or_path:
        raise ValueError("image_file_or_path not specified")
    elif op.isdir(args.image_file_or_path):
        for filename in os.listdir(args.image_file_or_path):
            if filename.endswith(".png") or filename.endswith(".jpg") and "pred" not in filename:
                image_list.append(args.image_file_or_path+"/"+filename)
    else:
        raise ValueError("Cannot find images at {}".format(args.image_file_or_path))

    run_inference(args, image_list, _gaze_bert)



if __name__ == "__main__":
    args = parse_args()
    main(args)
