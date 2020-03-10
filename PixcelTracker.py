import os
import copy
import queue
import argparse
import scipy.misc
import numpy as np
import math
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Pytorch
import torch
import torch.nn as nn
from torchvision.utils import save_image

import numpy as np

# Customized libraries
from libs.test_utils import *
from libs.model import transform
from libs.vis_utils import norm_mask
from libs.model import Model_switchGTfixdot_swCC_Res as Model
from libs.track_utils import seg2bbox, draw_bbox, match_ref_tar
from libs.track_utils import squeeze_all, seg2bbox_v2, bbox_in_tar_scale


from Tracker import JTSL
from Observer import Tracking_Observer
from utils import convert_pallate,read_txt_gtV2,calculate_iou

#from test_MOT_tracker import forward,adjust_bbox,bbox_next_frame,vis_bbox

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type = int, default = 1,
                        help = "batch size")
    parser.add_argument("-o","--out_dir",type = str,default = "results/",
                        help = "output saving path")
    parser.add_argument("--device", type = int, default = 5,
                        help = "0~4 for single GPU, 5 for dataparallel.")
    parser.add_argument("-c","--checkpoint_dir",type = str,
                        default = "weights/checkpoint_latest.pth.tar",
                        help = "checkpoints path")
    parser.add_argument("-s", "--scale_size", default=[360],
                        help = "scale size, either a single number for short edge, or a pair for height and width")
    parser.add_argument("--pre_num", type = int, default = 7,
                        help = "preceding frame numbers")
    parser.add_argument("--temp", type = float,default = 1,
                        help = "softmax temperature")
    parser.add_argument("--topk", type = int, default = 1,
                        help = "accumulate label from top k neighbors")
    parser.add_argument("-d", "--mot_dir", type = str,
                        default = "../dataset/MOT17Det/train",
                        help = "davis dataset path")
    parser.add_argument("--track_interval", type = int, default = 5,
                        help = "interval using  gt_bbox")


    print("Begin parser arguments.")
    args = parser.parse_args()
    args.is_train = False
    
    args.multiGPU = args.device == 5
    if not args.multiGPU:
        torch.cuda.set_device(args.device)
    #args.davis_dir = os.path.join(args.davis_dir, "JPEGImages/480p/")

    return args





if(__name__ == '__main__'):
    args = parse_args()
    model = Model(pretrainRes=False, temp = args.temp, uselayer=4)
    if(args.multiGPU):
        model = nn.DataParallel(model)
    checkpoint = torch.load(args.checkpoint_dir)
    best_loss = checkpoint['best_loss']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{} ({})' (epoch {})"
          .format(args.checkpoint_dir, best_loss, checkpoint['epoch']))
    model.cuda()
    model.eval()

    MOT_num="MOT17-02"
    video_dir=os.path.join(MOT_num,"img1")
    video_dir=os.path.join(args.mot_dir,video_dir)
    txt_dir=os.path.join(MOT_num,"gt/gt.txt")
    frames_gt = read_txt_gtV2(txt_dir)

    frame_list=read_frame_list(video_dir)

    track_id=0#trackする人物

    Tracker=JTSL(frames_gt,track_id,args.track_interval,model)
    Observer=Tracking_Observer(frame_list)

    Tracker.interface(frame_list,17,4,initial_box=None)
    for i in tqdm(range(1,500)):
        bbox_list=[]
        pop_list=[]
        if i==1:
            for n,initial_box in enumerate(frames_gt["1"]):
                os.makedirs("save_image/image/track_id_{}".format(n+1), exist_ok=True)
                os.makedirs("save_image/mask/track_id_{}".format(n+1), exist_ok=True)
                bbox=Tracker.interface(frame_list,i,n+1,initial_box)
                if bbox is None:
                    continue
                bbox=[n+1,int(bbox[1]),int(bbox[2]),int(bbox[3]),int(bbox[4])]
                bbox_list.append(bbox)
        else:
            for j in range(len(Observer.Tracklet[i-1])):
                n=Observer.Tracklet[i-1][j][0]
                bbox=Tracker.interface(frame_list,i,n,initial_box=None)
                if bbox is None:
                    pop_list.append(j)
                else:
                    bbox=[n,int(bbox[1]),int(bbox[2]),int(bbox[3]),int(bbox[4])]
                    bbox_list.append(bbox)

        Observer.Tracklet[i]=bbox_list
                    #break
        #for k in pop_list:
         #   Observer.Tracklet[i]
        Observer.save_txt(i)
        if i>=15:
            Observer.make_video()
            break
"""
    for i in tqdm(range(1,100)):
        try:
            if i==1:
                for n,initial_box in enumerate(frames_gt["1"]):
                    os.makedirs("save_image/image/track_id_{}".format(n+1), exist_ok=True)
                    os.makedirs("save_image/mask/track_id_{}".format(n+1), exist_ok=True)
                    bbox=Tracker.interface(frame_list,i,n+1,initial_box)
                    Observer.append_track(i,n+1,bbox)
            else:
                for n,initial_box in enumerate(frames_gt["1"]):
                    bbox=Tracker.interface(frame_list,i,n+1,initial_box=None)
                    Observer.append_track(i,n+1,bbox)
        except:
            #if i>=15:
            Observer.make_video()
            break
"""