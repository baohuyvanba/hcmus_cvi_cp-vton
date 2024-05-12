#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os

from tqdm import tqdm
from cp_dataset import CPDataset, CPDataLoader                                      #file: cp_dataset.py
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint  #file: networks.py

from torch.utils.tensorboard import SummaryWriter
from visualization import board_add_image, board_add_images                         #file: visualization.py

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
gpus_id = int(os.environ["LOCAL_RANK"])

def get_opt():
    """
    This function defines the command-line arguments for the program.
    Returns: A namespace object (opt) containing all the parsed arguments and their values.
    """

    #Define arguments related to model and training: name, batch size, number of workers
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument('-j', '--workers', type=int, default=6)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument("--stage", default="GMM")
    #Define arguments related to data
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    #Define arguments related to image processing
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    # Define arguments related to training optimization
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--save_count", type=int, default=100)
    #Define arguments related to data training visualization and logging
    parser.add_argument("--display_count", type=int, default = 20)
    #Shuffling data
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def train_gmm(opt, train_loader, model, board):
    """
    This function run the train processs for the GMM model on the provided data loader get from CP-VTON dataset.

    Args:
        opt (argparse.Namespace): Namespace containing the parsed command-line arguments.
        train_loader (DataLoader): Data loader for training data.
        model (nn.Module): The GMM model to be trained.
        board (object): Object for logging information (e.g., TensorBoard).
    """
    #Create model and move to GPU: Distributed Data Parallel (DDP) for multi-GPU training
    model = DDP(model.to(gpus_id), [gpus_id])
    model.train()

    #Define loss function: L1 Loss
    criterionL1 = nn.L1Loss()
    #Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    #Training loop process
    for epoch in tqdm(range(opt.keep_step + opt.decay_step)):
        #Set epoch for the data sampler
        train_loader.data_loader.sampler.set_epoch(epoch)
        #Get a batch of data from dataset
        inputs = train_loader.next_batch()

        #Move data to GPUs for training
        image        = inputs['image'].to(gpus_id)      # [4, 3, 256, 192]
        img_pose     = inputs['pose_image'].to(gpus_id) # [4, 1, 256, 192]
        img_headmsk  = inputs['head'].to(gpus_id)       # [4, 3, 256, 192]
        person_shape = inputs['shape'].to(gpus_id)      # [4, 1, 256, 192]
        agnostic     = inputs['agnostic'].to(gpus_id)   # [4, 22, 256, 192]
        cloth        = inputs['cloth'].to(gpus_id)      # [4, 3, 256, 192]
        cthmask      = inputs['cloth_mask'].to(gpus_id) # [4, 1, 256, 192]
        img_cthmask  = inputs['parse_cloth'].to(gpus_id)# [4, 3, 256, 192] #ground_truth for warp cloth mask
        img_grid     = inputs['grid_image'].to(gpus_id) # [4, 3, 256, 192]

        #Forward pass through the model
        grid, theta = model(agnostic, cloth)

        #Warp cloth and mask using the predicted grid, warped_grid use for visualization
        warped_cloth = F.grid_sample(cloth, grid, padding_mode='border', align_corners=True)
        warped_mask = F.grid_sample(cthmask, grid, padding_mode='zeros', align_corners=True)
        warped_grid = F.grid_sample(img_grid, grid, padding_mode='zeros', align_corners=True)

        #Visualizations for logging
        visuals = [ [img_headmsk, person_shape, img_pose],         #head, shape, pose
                   [cloth, warped_cloth, img_cthmask],             #cloth, warped cloth, parsed cloth images
                   [warped_grid, (warped_cloth+image)*0.5, image]] #warped grid, combine image with warped_cloth, input image

        #Calculate loss with L1 Loss
        loss = criterionL1(warped_cloth, img_cthmask)
        #Backpropagation and optimization
        optimizer.zero_grad()  #Clear gradients
        loss.backward()        #Backpropagate loss
        optimizer.step()       #Update model weights

        #Logging (every number of steps, which was defined with argument before)
        if (epoch+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, epoch+1)
            board.add_scalar('metric', loss.item(), epoch+1)
        #Save weight after number of steps
        if (epoch+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (epoch+1)))


def train_tom(opt, train_loader, model, board):
    model = DDP(model.to(gpus_id), [gpus_id])
    model.train()
    
    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    for step in tqdm(range(opt.keep_step + opt.decay_step)):
        train_loader.data_loader.sampler.set_epoch(step)
        inputs = train_loader.next_batch()
            
        im = inputs['image'].to(gpus_id)
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].to(gpus_id)
        c = inputs['cloth'].to(gpus_id)
        cm = inputs['cloth_mask'].to(gpus_id)
        
        outputs = model(torch.cat([agnostic, c],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        p_tryon = c * m_composite+ p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose], 
                   [c, cm*2-1, m_composite*2-1], 
                   [p_rendered, p_tryon, im]]
            
        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, cm)
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            board.add_scalar('L1', loss_l1.item(), step+1)
            board.add_scalar('VGG', loss_vgg.item(), step+1)
            board.add_scalar('MaskL1', loss_mask.item(), step+1)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))



def main():
    opt = get_opt()
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)

    writer = SummaryWriter(os.path.join(opt.tensorboard_dir, opt.name))
    
    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model, writer)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, model, writer)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
        
  
    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
