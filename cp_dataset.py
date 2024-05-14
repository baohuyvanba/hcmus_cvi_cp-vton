# coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DistributedSampler

from PIL import Image
from PIL import ImageDraw

import os
import numpy as np
import json


class CPDataset(data.Dataset):
    """
    Dataset class for input (CP-VTON dataset)
    """
    def __init__(self, opt):
        super(CPDataset, self).__init__()
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode
        self.stage = opt.stage
        self.data_list = f"{opt.datamode}_pairs.txt"
        self.fine_height = opt.fine_height
        self.fine_width  = opt.fine_width
        self.radius = opt.radius
        self.data_path = os.path.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize( #Normalize: (Each value of tensor - mean)/std
                mean=(0.5,),
                std=(0.5,)
            )
        ])

        #read data list
        img_names = []
        cth_names = []

        with open(os.path.join(opt.dataroot, self.data_list), 'r') as f:
            for line in f.readlines():
                #Form: image_1_name.txt cloth_1_name.txt ...
                img_name, cth_name = line.strip().split()
                img_names.append(img_name)
                cth_names.append(cth_name)

        self.img_names = img_names
        self.cth_names = cth_names

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        cth_name = self.cth_names[index] #Cloth image's name
        img_name = self.img_names[index] #Person image's name

        #Note - Pytorch image: (channel x height x width)

        #CLOTH and CLOTH Mask in GMM and TOM
        if self.stage == 'GMM':
            cloth = Image.open(os.path.join(self.data_path, 'cloth', cth_name))
            clothmask = Image.open(os.path.join(self.data_path, 'cloth-mask', cth_name))
        else: #self.stage == 'TOM'
            cloth = Image.open(os.path.join(self.data_path, 'warped-cloth', cth_name))
            clothmask = Image.open(os.path.join(self.data_path, 'warped-cloth-mask', cth_name))

        #CLOTH: read
        #cloth = Image.open(os.path.join(self.data_path, 'cloth', cth_name))
        cloth = self.transform(cloth)
        if (cloth.shape[1], cloth.shape[2]) != (256, 192):
            cloth = transforms.Resize((256, 192))(cloth) #256x192x3 -> 3x256x192


        #CLOTH-MASK: read
        #clothmask = Image.open(os.path.join(self.data_path, 'cloth-mask', cth_name))
                    #This mask image only has 1 color channel
        cmask_array = np.array(clothmask)                       #1x256x192 -> 256x192
        cmask_array = (cmask_array >= 128).astype(np.float32)   #Binary mask: 0(background), 1(foreground/cloth region)
        clothmask = torch.from_numpy(cmask_array).unsqueeze(0)  #256x192 -> 1x256x192: add dim at 0 using unsqueeze because Pytorch image data is (channel x height x width)


        #IMAGE: read person image
        image = Image.open(os.path.join(self.data_path, 'image', img_name))
        image = self.transform(image) #256x192x3 -> 3x256x192 and normalize [0, 255] to [-1, 1]


        #IMAGE-PARSE: read parsing image
        parse_name = img_name.replace('.jpg', '.png')
        img_parse = Image.open(os.path.join(self.data_path, 'image-parse', parse_name))
                    #This is RGB image: 3x256x192
        parse_array = np.array(img_parse) #3x256x192
          #Binary shape parse: 1:all body, 0:background
        parse_shape = (parse_array > 0)
          #downsample shape parse
        parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.fine_width // 16, self.fine_height // 16))
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height))
        person_shape = self.transform(parse_shape)     #to Tensor (1x256x192) then Normalize: [0,255] -> [-1,1]

          #Binary head mask: 1:head segment -> keep image identity
        parse_head = (parse_array == 1).astype(np.float32) + \
                     (parse_array == 2).astype(np.float32) + \
                     (parse_array == 4).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32)
        person_head = torch.from_numpy(parse_head)      #to Tensor: {0,1}

          #Binary cloth mask: 1:upbody segment -> where to add cloth segment
        parse_cloth = (parse_array == 5).astype(np.float32) + \
                      (parse_array == 6).astype(np.float32) + \
                      (parse_array == 7).astype(np.float32)
        person_cthmask = torch.from_numpy(parse_cloth)  #to Tensor: {0,1}

          #DIFF: Binary rest of body mask (Neither cloth region nor background)
        restbody_mask = (parse_array == 1).astype(np.float32) + \
                        (parse_array == 2).astype(np.float32) + \
                        (parse_array == 3).astype(np.float32) + \
                        (parse_array == 4).astype(np.float32) + \
                        (parse_array > 7).astype(np.float32)
        restbody_mask = torch.from_numpy(restbody_mask) #to Tensor: {0, 1}

        #Upper cloth: cloth region and  head region mask on image
        img_cthmask = image * person_cthmask + (1 - person_cthmask)  #Get pixels value of cloth region and fill 1 for other parts -> [-1, 1]*{0,1} + {1, 0} = [-1, 1]: Mask of cloth region on person image
        img_headmsk = image * person_head - (1 - person_head)        #Get pixels value of head region and fill 0 for other parts  -> [-1, 1]*{0,1} + {0, 1} = [-1, 1]: Mask of head region on person image

        #POSE: read
        pose_name = img_name.replace('.jpg', '_keypoints.json')
        with open(os.path.join(self.data_path, 'pose', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        img_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(img_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
                pose_draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
            one_map = self.transform(one_map)
            pose_map[i] = one_map[0]

        # just for visualization
        img_pose = self.transform(img_pose)

        #PERSON REPRESENTATION: Cloth-agnostic Representation: person shape, head region mask, pose map.
        agnostic = torch.cat([person_shape, img_headmsk, pose_map], 0)


        im_g = Image.open('grid.png')
        im_g = self.transform(im_g)

        result = {
            'c_name': cth_name,         # for visualization
            'im_name': img_name,        # for visualization or ground truth
            'cloth': cloth,             # for input
            'cloth_mask': clothmask,    # for input
            'image': image,             # for visualization
            'agnostic': agnostic,       # for input
            'parse_cloth': img_cthmask, # for ground truth
            'shape': person_shape,      # for visualization
            'head': img_headmsk,        # for visualization
            'restbody': restbody_mask,  #
            'pose_image': img_pose,     # for visualization
            'grid_image': im_g,         # for visualization
        }

        return result


class CPDataLoader(object):
    """
    This class creates a data loader for cp-vton dataset.

    Attributes:
        dataset (torch.utils.data.Dataset): The dataset to be loaded.
        data_loader (torch.utils.data.DataLoader): The data loader object.
        data_iter (iter): An iterator over the data loader.
    """
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()
        self.dataset = dataset
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,     #Define the batch size
            num_workers=opt.workers,       #Define the number of worker threads to use for data loading (main thread only)
            pin_memory=True,               #Improve data transfer's speed between CPUs and GPUs
            sampler=DistributedSampler(dataset)
        )
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        """
        Returns the next batch of data from the data loader.
        Returns:
            batch: A batch of data from the dataset.
        """
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()
        return batch