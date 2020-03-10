import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import convert_pallate,read_txt_gtV2,calculate_iou

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


from libs.test_utils import *
from libs.model import transform
from libs.vis_utils import norm_mask
from libs.model import Model_switchGTfixdot_swCC_Res as Model
from libs.track_utils import seg2bbox, draw_bbox, match_ref_tar
from libs.track_utils import squeeze_all, seg2bbox_v2, bbox_in_tar_scale


class JTSL():
    def __init__(self,frames_gt,track_id,track_interval,model,scale_size=[360]):
        self.model=model
        self.scale_size=scale_size
        self.track_interval=track_interval
        self.frames_gt=frames_gt



    def interface(self,frame_list,i,id_num,initial_box):
        if initial_box!=None:
            self.bbox_to_mask(initial_box,1,id_num)
            first_mask=Image.open("save_image/mask/track_id_{0}/{1:04d}.png".format(id_num,1))
        else:
            first_mask=Image.open("save_image/mask/track_id_{0}/{1:04d}.png".format(id_num,i-1))
        self.large_seg, self.first_seg, self.seg_ori = read_seg(first_mask, self.scale_size)
        self.first_bbox = seg2bbox(self.large_seg, margin=0.1) #class数分だけbounding boxを指すクラスが作られる

        for k,v in self.first_bbox.items(): #Bboxの大きさを0.125倍する=1/8
            v.upscale(0.125)

        transforms=create_transforms()
        frame1, ori_h, ori_w = read_frame(frame_list[i-1], transforms,self.scale_size)
        n, c, h, w = frame1.size()
        coords = self.first_seg[0,1].nonzero()
        coords = coords.flip(1)


        tar_frame,ori_h,ori_w=read_frame(frame_list[i],transforms,self.scale_size)

        with torch.no_grad():
            bbox_pre=self.first_bbox
            framei=frame1
            segi=self.first_seg
            _, segi_int = torch.max(segi, dim=1)
            segi = to_one_hot(segi_int)


            bbox_tar,coords_ref_tar = self.bbox_next_frame(framei, segi,tar_frame,bbox_pre,self.model)

            if len(bbox_tar)<2:
                return None
            tmp = copy.deepcopy(bbox_tar[1])
            tmp.upscale(8)

            bbox=self.vis_bbox(tar_frame, tmp, os.path.join("result", 'track', 'frame'+'{0:04d}'.format(i)+'.png'), coords_ref_tar[1], segi[0,1,:,:])
            gt_bbox=[]
            #if i%self.track_interval==0:
        gt_bbox=self.appropriate_box(bbox,i)
        #print(gt_bbox)

        if len(gt_bbox)==0:
            self.bbox_to_mask(bbox,i,id_num)
            print(i,id_num,"not_iou")
            return None
            #print("Y")
        else:
            self.bbox_to_mask(gt_bbox[-1],i,id_num)
            return gt_bbox[-1]
            #print("N")


    def appropriate_box(self,bbox,i):
        gt_bbox=[]
        max_iou=0
        for j in range(len(self.frames_gt[str(i+1)])):
            iou=calculate_iou(bbox,self.frames_gt[str(i+1)][j])
            #print(iou)
            if iou>0.5 and iou>max_iou:
                gt_bbox.append(self.frames_gt[str(i+1)][j])
                max_iou=iou
        
        return gt_bbox
    

    def  bbox_next_frame(self,img_ref, seg_ref, img_tar,bbox_ref,model):
        F_ref, F_tar = self.forward(img_ref, img_tar, self.model, seg_ref, return_feature=True) #reference targe画像をそれぞれencodeしたfeature
        seg_ref = seg_ref.squeeze(0)
        F_ref, F_tar = squeeze_all(F_ref, F_tar)
        c, h, w = F_ref.size()
	# get coordinates of each point in the target frame
        coords_ref_tar = match_ref_tar(F_ref, F_tar, seg_ref, 1)

        bbox_tar = bbox_in_tar_scale(coords_ref_tar, bbox_ref, h, w)

        return bbox_tar,coords_ref_tar


    def forward(self,frame1, frame2, model, seg, return_feature=False):
        n, c, h, w = frame1.size()
        frame1_gray = frame1[:,0].view(n,1,h,w)
        frame2_gray = frame2[:,0].view(n,1,h,w)
        frame1_gray = frame1_gray.repeat(1,3,1,1)
        frame2_gray = frame2_gray.repeat(1,3,1,1)


        output = model(frame1_gray, frame2_gray, frame1, frame2)
        if(return_feature):
            return output[-2], output[-1]
        aff = output[2]
        frame2_seg = transform_topk(aff,seg.cuda(),k=args.topk)
    
        return frame2_seg

    def convert_pallate(self,image):
        palette=image.getpalette()
        #print(np.array(palette))
        palette = np.array(palette).reshape(-1, 3)
        palette[0,:]=0
        palette[1,:]=(0,128,0)
        palette[2,:]=(128,0,0)
        palette[3:,:]=(0,0,0)
        palette = palette.reshape(-1).tolist()
        image.putpalette(palette)
        seg=image

        return seg

    def vis_bbox(self,im, bbox, name, coords, seg):
        im = im * 128 + 128
        im = im.squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_LAB2BGR)
        fg_idx = seg.nonzero()
        im = draw_bbox(im, bbox, (0,0,255))
        coordsx_list=[]
        coordsy_list=[]
        for cnt in range(coords.size(0)):
            coord_i = coords[cnt]
            cv2.circle(im, (int(coord_i[0]*8), int(coord_i[1]*8)), 2, (0,255,0), thickness=-1)
            coordsx_list.append(int(coord_i[0]*8))
            coordsy_list.append(int(coord_i[1]*8))
        xmax,xmin=max(coordsx_list),min(coordsx_list)
        ymax,ymin=max(coordsy_list),min(coordsy_list)
        cv2.imwrite(name, im)
        return (1,xmin*3-10,ymin*3-10,xmax*3+10,ymax*3+10)

    def bbox_to_mask(self,bbox,i,id_num):
        bx1,by1,bx2,by2=bbox[1],bbox[2],bbox[3],bbox[4]
        bx1,by1,bx2,by2=int(bx1),int(by1),int(bx2),int(by2)
        first_mask=np.zeros((1080,1920))
        first_mask[by1:by2,bx1:bx2]=1
        first_mask=Image.fromarray(np.uint8(first_mask)).convert("P")
        first_mask=self.convert_pallate(first_mask)
        large_seg, first_seg, seg_ori = read_seg(first_mask, self.scale_size)
        imwrite_indexed("save_image/mask/track_id_{0}/{1:04d}.png".format(id_num,i), seg_ori)


    def adjust_bbox(self,bbox_now, bbox_pre, a, h, w):
        for cnt in bbox_pre.keys():
            if(cnt == 0):
                continue
            if(cnt in bbox_now and bbox_pre[cnt] is not None and bbox_now[cnt] is not None):
                bbox_now_h = (bbox_now[cnt].top  + bbox_now[cnt].bottom) / 2.0
                bbox_now_w = (bbox_now[cnt].left + bbox_now[cnt].right) / 2.0
            
                bbox_now_height_ = bbox_now[cnt].bottom - bbox_now[cnt].top
                bbox_now_width_  = bbox_now[cnt].right  - bbox_now[cnt].left

                bbox_pre_height = bbox_pre[cnt].bottom - bbox_pre[cnt].top
                bbox_pre_width  = bbox_pre[cnt].right  - bbox_pre[cnt].left
                bbox_now_height = a * bbox_now_height_ + (1 - a) * bbox_pre_height
                bbox_now_width  = a * bbox_now_width_  + (1 - a) * bbox_pre_width
                bbox_now[cnt].left   = math.floor(bbox_now_w - bbox_now_width / 2.0)
                bbox_now[cnt].right  = math.ceil(bbox_now_w + bbox_now_width / 2.0)
                bbox_now[cnt].top = math.floor(bbox_now_h - bbox_now_height / 2.0)
                bbox_now[cnt].bottom = math.ceil(bbox_now_h + bbox_now_height / 2.0)
                bbox_now[cnt].left = max(0, bbox_now[cnt].left)
                bbox_now[cnt].right = min(w, bbox_now[cnt].right)
                bbox_now[cnt].top = max(0, bbox_now[cnt].top)
                bbox_now[cnt].bottom = min(h, bbox_now[cnt].bottom)
            
            return bbox_now

