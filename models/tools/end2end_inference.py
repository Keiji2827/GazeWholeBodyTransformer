"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

End-to-end inference codes for 
3D human body mesh reconstruction from an image

python ./metro/tools/end2end_inference_bodymesh.py 
       --resume_checkpoint ./models/metro_release/metro_3dpw_state_dict.bin
       --image_file_or_path ./samples/human-body
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import code
import json
import copy
import time
import datetime
import torch
import torchvision.models as models
from torchvision.utils import make_grid
import numpy as np
import cv2
from torch.utils.data import Subset, DataLoader
from models.bert.modeling_bert import BertConfig
from models.bert.modeling_metro import METRO_Body_Network as METRO_Network
from models.bert.modeling_metro import METRO
from models.bert.modeling_gabert import GAZEFROMBODY
from models.smpl._smpl import SMPL, Mesh
from models.hrnet.hrnet_cls_net_featmaps import get_cls_net
from models.hrnet.config import config as hrnet_config
from models.hrnet.config import update_config as hrnet_update_config
from models.dataloader.gafa_loader import create_gafa_dataset
from models.utils.logger import setup_logger
from models.utils.metric_logger import AverageMeter
from models.utils.miscellaneous import mkdir, set_seed

from PIL import Image
from torchvision import transforms

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


def save_checkpoint(model, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))

    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save(model_to_save, op.join(checkpoint_dir, 'model.bin'))
            torch.save(model_to_save.state_dict(), op.join(checkpoint_dir, 'state_dict.bin'))
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir




class CosLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        outputs = outputs.reshape(-1, outputs.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])
        cos =  torch.sum(outputs*targets,dim=-1)
        cos[cos > 1] = 1
        cos[cos < -1] = -1
        rad = torch.acos(cos)
        loss = torch.rad2deg(rad)#0.5*(1-cos)#criterion(pred_gaze,gaze_dir)

        return loss

def run(args, train_dataloader, val_dataloader, _gaze_network, smpl, mesh_sampler):

    max_iter = len(train_dataloader)
    print("len of dataset:",max_iter)
    epochs = args.num_train_epochs

    optimizer = torch.optim.Adam(params=list(_gaze_network.parameters()),lr=args.lr,
                                            betas=(0.9, 0.999), weight_decay=0) 

    start_training_time = time.time()
    end = time.time()
    _gaze_network.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()

    criterion_mse = CosLoss().cuda(args.device)

    for epoch in range(epochs):
        for iteration, batch in enumerate(train_dataloader):

            iteration += 1
            #epoch = iteration
            _gaze_network.train()

            image = batch["image"].cuda(args.device)
            gaze_dir = batch["gaze_dir"].cuda(args.device)
            head_dir = batch["head_dir"].cuda(args.device)
            body_dir = batch["body_dir"].cuda(args.device)
            head_pos = batch["head_pos"].cuda(args.device)
            keypoints = batch["keypoints"].cuda(args.device)

            batch_imgs = image
            batch_size = image.size(0)


            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
            data_time.update(time.time() - end)


            # 指定した位置にサイズ1の次元を挿入する unsqeeze()
            #print(batch_imgs.shape)
            #print("gaze_dir:",gaze_dir)

            # forward-pass
            direction = _gaze_network(batch_imgs, smpl, mesh_sampler, head_dir)
            #print(direction.shape)

            loss = criterion_mse(direction,gaze_dir).mean()

            # update logs
            log_losses.update(loss.item(), batch_size)

            # back prop
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if(iteration%args.logging_steps==0):
                #print("iteration:",iteration)
                eta_seconds = batch_time.avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(
                    ' '.join(
                    ['eta: {eta}', 'epoch: {ep}', 'iter: {iter}',]
                    ).format(eta=eta_string, ep=epoch, iter=iteration) 
                    + ":loss:{:.4f}, lr:{:.6f}".format(log_losses.avg, optimizer.param_groups[0]["lr"])
                )
        
        val = run_validate(args, val_dataloader, 
                            _gaze_network, 
                            criterion_mse,
                            smpl,
                            mesh_sampler)
        print("val:", val)
        checkpoint_dir = save_checkpoint(_gaze_network, args, epoch, iteration)
        print("save trained model at ", checkpoint_dir)

def run_validate(args, val_dataloader, _metro_network, criterion_mse, smpl,mesh_sampler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mse = AverageMeter()

    _metro_network.eval()
    smpl.eval()

    with torch.no_grad():        
        for iteration, batch in enumerate(val_dataloader):
            iteration += 1
            epoch = iteration

            image = batch["image"].cuda(args.device)
            gaze_dir = batch["gaze_dir"].cuda(args.device)

            batch_imgs = image
            batch_size = image.size(0)

            # forward-pass
            direction = _metro_network(batch_imgs, smpl, mesh_sampler, gaze_dir)
            #print(direction.shape)

            loss = criterion_mse(direction,gaze_dir).mean()

            # update logs
            mse.update(loss.item(), batch_size)

            #if (iteration > 1000):
            #    break


    return mse.avg


def run_inference(args, image_list, _metro_network, smpl, mesh_sampler):
    # switch to evaluate mode
    # train modeだと.train()となる
    _metro_network.eval()
    
    for image_file in image_list:
        if 'pred' not in image_file:
            att_all = []
            print("in inference",image_file)
            img = Image.open(image_file)
            # from torchvision import transforms
            img_tensor = transform(img)
            img_visual = transform_visualize(img)

            # 指定した位置にサイズ1の次元を挿入する unsqeeze()
            batch_imgs = torch.unsqueeze(img_tensor, 0).cuda()
            batch_visual_imgs = torch.unsqueeze(img_visual, 0).cuda()
            # 推論はここで完了
            # forward-pass
            pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices, hidden_states, att = _metro_network(batch_imgs, smpl, mesh_sampler)
                
            return
            # obtain 3d joints from full mesh
            pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)

            pred_3d_pelvis = pred_3d_joints_from_smpl[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
            pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:]
            pred_3d_joints_from_smpl = pred_3d_joints_from_smpl - pred_3d_pelvis[:, None, :]
            pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]

            # save attantion
            att_max_value = att[-1]
            att_cpu = np.asarray(att_max_value.cpu().detach())
            att_all.append(att_cpu)

            # obtain 3d joints, which are regressed from the full mesh
            pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)
            pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:]
            # obtain 2d joints, which are projected from 3d joints of smpl mesh
            pred_2d_joints_from_smpl = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
            pred_2d_431_vertices_from_smpl = orthographic_projection(pred_vertices_sub2, pred_camera)
            visual_imgs_att = visualize_mesh_and_attention( renderer, batch_visual_imgs[0],
                                                        pred_vertices[0].detach(), 
                                                        pred_vertices_sub2[0].detach(), 
                                                        pred_2d_431_vertices_from_smpl[0].detach(),
                                                        pred_2d_joints_from_smpl[0].detach(),
                                                        pred_camera.detach(),
                                                        att[-1][0].detach())

            visual_imgs = visual_imgs_att.transpose(1,2,0)
            visual_imgs = np.asarray(visual_imgs)
                    
            temp_fname = image_file[:-4] + '_metro_pred.jpg'
            print('save to ', temp_fname)
            cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))

    return 




def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--image_file_or_path", default='./test_images/human-body', type=str, 
                        help="test data")
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='models/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for inference.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--per_gpu_train_batch_size", default=30, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=30, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--lr', "--learning_rate", default=1e-4, type=float, 
                        help="The initial lr.")
    parser.add_argument("--num_train_epochs", default=20, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--drop_out", default=0.1, type=float, 
                        help="Drop out ratio in BERT.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                        help="The Image Feature Dimension.")          
    parser.add_argument("--hidden_feat_dim", default='1024,256,128', type=str, 
                        help="The Image Feature Dimension.")   
    parser.add_argument("--legacy_setting", default=True, action='store_true',)
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--run_eval_only", default=False, action='store_true',) 
    parser.add_argument('--logging_steps', type=int, default=10000, 
                        help="Log every X steps.")
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")


    args = parser.parse_args()
    return args

# 最初はここから
def main(args):
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    # 並列処理の設定
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)

    # default='output/'
    #mkdir(args.output_dir)
    logger = setup_logger("METRO Inference", args.output_dir, 0)
    # randomのシード
    # default=88
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    # from metro.modeling._smpl import SMPL, Mesh
    mesh_smpl = SMPL().to(args.device)
    mesh_sampler = Mesh()

    # Renderer for visualization
    # from metro.utils.renderer import Renderer, visualize_reconstruction, visualize_reconstruction_test, visualize_reconstruction_no_text, visualize_reconstruction_and_att_local
    #renderer = Renderer(faces=mesh_smpl.faces.cpu().numpy())

    # Load pretrained model
    # --resume_checkpoint ./models/metro_release/metro_3dpw_state_dict.bin
    logger.info("Inference: Loading from checkpoint {}".format(args.resume_checkpoint))

    if args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
        # この中っぽいとは思う。
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _metro_network = torch.load(args.resume_checkpoint)
    else:
        # どうやらこっち側みたい
        # Build model from scratch, and load weights from state_dict.bin
        trans_encoder = []
        # input_feat_dim default='2051,512,128'
        input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
        # hidden_feat_dim default='1024,256,128'
        hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
        output_feat_dim = input_feat_dim[1:] + [3]
        # init three transformer encoders in a loop
        # transformerの初期化
        # output_feat_dim = [512, 128, 3]
        for i in range(len(output_feat_dim)):
            # from metro.modeling.bert import BertConfig, METRO
            config_class, model_class = BertConfig, METRO
            # default='metro/modeling/bert/bert-base-uncased/'
            config = config_class.from_pretrained(args.model_name_or_path)

            config.output_attentions = False
            config.img_feature_dim = input_feat_dim[i] 
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]

            if args.legacy_setting==True:
                # During our paper submission, we were using the original intermediate size, which is 3072 fixed
                # We keep our legacy setting here 
                args.intermediate_size = -1
            else:
                # We have recently tried to use an updated intermediate size, which is 4*hidden-size.
                # But we didn't find significant performance changes on Human3.6M (~36.7 PA-MPJPE)
                args.intermediate_size = int(args.hidden_size*4)

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            # model_class = METRO
            model = model_class(config=config) 
            logger.info("Init model from scratch.")
            trans_encoder.append(model)

        # for ここまで
        # init ImageNet pre-trained backbone model
        # arch default='hrnet-w64'
        if args.arch=='hrnet':
            hrnet_yml = 'models/hrnet/weights/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = './models/hrnet/weights/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
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

        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info('Transformers total parameters: {}'.format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        # build end-to-end METRO network (CNN backbone + multi-layer transformer encoder)
        # from metro.modeling.bert import METRO_Body_Network as METRO_Network
        # ここでモデルの初期化
        _metro_network = METRO_Network(args, config, backbone, trans_encoder, mesh_sampler)

        logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
        cpu_device = torch.device('cpu')
        state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
        _metro_network.load_state_dict(state_dict, strict=False)
        del state_dict
    # model の構築ここまで

    # update configs to enable attention outputs
    setattr(_metro_network.trans_encoder[-1].config,'output_attentions', True)
    setattr(_metro_network.trans_encoder[-1].config,'output_hidden_states', True)
    _metro_network.trans_encoder[-1].bert.encoder.output_attentions = True
    _metro_network.trans_encoder[-1].bert.encoder.output_hidden_states =  True
    for iter_layer in range(4):
        _metro_network.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
    for inter_block in range(3):
        setattr(_metro_network.trans_encoder[-1].config,'device', args.device)

    _metro_network.to(args.device)
    backbone_train = copy.deepcopy(backbone)


    logger.info("Run inference")

    _gaze_network = GAZEFROMBODY(args, _metro_network, backbone_train)
    _gaze_network.to(args.device)

    print(args.device)
    if args.device == 'cuda':
        print("distribution")
        _gaze_network = torch.nn.DataParallel(_gaze_network) # make parallel
        torch.backends.cudnn.benchmark = True

    if args.run_eval_only == True:
        image_list = []
        # --image_file_or_path ./samples/human-body
        if not args.image_file_or_path:
            raise ValueError("image_file_or_path not specified")
        # ファイルの場合
        if op.isfile(args.image_file_or_path):
            image_list = [args.image_file_or_path]
        # ディレクトリの場合
        elif op.isdir(args.image_file_or_path):
            # should be a path with images only
            for filename in os.listdir(args.image_file_or_path):
                if filename.endswith(".png") or filename.endswith(".jpg") and 'pred' not in filename:
                    image_list.append(args.image_file_or_path+'/'+filename) 
        else:
            raise ValueError("Cannot find images at {}".format(args.image_file_or_path))

        logger.info("Run eval only\nNot use")
        # 推論実行
        run_inference(args, image_list, _metro_network, mesh_smpl, mesh_sampler)    

    else:
        logger.info("Run train")
        exp_names = [
        'library/1026_3',
        'library/1028_2',
        'library/1028_5',
        'lab/1013_1',
        'lab/1014_1',
        'kitchen/1022_4',
        'kitchen/1015_4',
        'living_room/004',
        'living_room/005',
        'courtyard/004',
        'courtyard/005',
                    ]
        dset = create_gafa_dataset(exp_names=exp_names)
        #train_idx, val_idx = np.arange(0, 800), np.arange(int(len(dset)*0.9), len(dset))
        train_idx, val_idx = np.arange(0, int(len(dset)*0.9)), np.arange(int(len(dset)*0.9), len(dset))
        train_dset = Subset(dset, train_idx)
        val_dset   = Subset(dset, val_idx)

        train_dataloader = DataLoader(
            train_dset, batch_size=20, num_workers=4, pin_memory=True, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dset, batch_size=10, shuffle=False, num_workers=4, pin_memory=True
        )
        # Training
        run(args, train_dataloader, val_dataloader, _gaze_network, mesh_smpl, mesh_sampler)

if __name__ == "__main__":
    args = parse_args()
    main(args)
