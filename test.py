#coding=utf-8
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
from tqdm import tqdm
import time
from cp_dataset import CPDataset, CPDataLoader
from FASCODE_IMAGE import FASCODE_IMAGE

from networks import GMM, UnetGenerator, load_checkpoint

from torch.utils.tensorboard import SummaryWriter
from visualization import board_add_image, board_add_images, save_images

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
gpus_id = dist.get_rank()

def get_opt():
    #Define arguments related to model and training: name, batch size, number of workers
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TOM")
    parser.add_argument('-j', '--workers', type=int, default=6)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
     #Define arguments related to data
    parser.add_argument("--dataroot", default = "data") #directory to data
    parser.add_argument("--datamode", default = "test")
    parser.add_argument("--stage", default = "TOM") # change to TOM or GMM
    parser.add_argument("--data_list", default = "test_pairs.txt") #txt includes list of images pairs for test
    
     #Define arguments related to image processing
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    
    #Define argument for saving outputs of the testing phase
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/TOM/tom_final.pth', help='model checkpoint for test')
    parser.add_argument("--display_count", type=int, default = 1)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def test_gmm(opt, test_loader, model, board):
    model = DDP(model.cuda(), [gpus_id])
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    # save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    save_dir = "data/test/"

    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')

    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)

    for step, inputs in tqdm(enumerate(test_loader.data_loader)):
        test_loader.data_loader.sampler.set_epoch(step)
        # iter_start_time = time.time()
        
        #Loading data to GPU for computing
        c_names = inputs['c_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
        
        #Forward the model to get the result      
        grid, theta = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border', align_corners=True)
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros', align_corners=True)
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros', align_corners=True)

        #Create visual board to visualize results
        visuals = [ [im_h, shape, im_pose], 
                   [c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im]]
        
        save_images(warped_cloth, c_names, warp_cloth_dir) 
        save_images(warped_mask*2-1, c_names, warp_mask_dir) 

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
        
def test_tom(opt, test_loader, model, board):
    model = DDP(model.cuda(), [gpus_id])
    model.eval()
    
    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    try_on_dir = os.path.join(save_dir, 'try-on')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)

    for step, inputs in tqdm(enumerate(test_loader.data_loader)):
         #Loading data to GPU for computing
        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        
        #Forwarding data to model and get the results
        outputs = model(torch.cat([agnostic, c],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        #Visualize the results in a 3x3 board
        visuals = [ [im_h, shape, im_pose], 
                   [c, 2*cm-1, m_composite], 
                   [p_rendered, p_tryon, im]]
            
        save_images(p_tryon, im_names, try_on_dir) 
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)

def main():
    opt = get_opt()
    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
   
    # create model & train
    if opt.stage == 'GMM':
        model = GMM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_gmm(opt, train_loader, model, board)
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_tom(opt, train_loader, model, board)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
  
    print('Finished test %s, named: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
