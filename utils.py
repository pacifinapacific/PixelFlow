from  PIL import Image
import numpy as np
import os
import csv
import copy
persons_class = ["1"]

def convert_pallate(mask_path):
    seg=Image.open(mask_path)
    image=seg.quantize()
    palette=image.getpalette()
    palette = np.array(palette).reshape(-1, 3)
    palette[0,:]=0
    palette[1,:]=(0,128,0)
    palette[2,:]=(128,0,0)
    palette[3:,:]=(0,0,0)
    palette = palette.reshape(-1).tolist()
    image.putpalette(palette)
    seg=image

    return seg
    

def xywh2xyxy(bbox):
    """
    convert bbox from [x,y,w,h] to [x1, y1, x2, y2]
    :param bbox: bbox in string [x, y, w, h], list
    :return: bbox in float [x1, y1, x2, y2], list
    """
    copy.deepcopy(bbox)
    bbox[0] = float(bbox[0])
    bbox[1] = float(bbox[1])
    bbox[2] = float(bbox[2]) + bbox[0]
    bbox[3] = float(bbox[3]) + bbox[1]

    return bbox
def reorder_frameID(frame_dict):
    """
    reorder the frames dictionary in a ascending manner
    :param frame_dict: a dict with key = frameid and value is a list of lists [object id, x, y, w, h] in the frame, dict
    :return: ordered dict by frameid
    """
    keys_int = sorted([int(i) for i in frame_dict.keys()])

    new_dict = {}
    for key in keys_int:
        new_dict[str(key)] = frame_dict[str(key)]
    return new_dict

def read_txt_gtV2(textpath):
    """
    read gt.txt to a dict
    :param textpath: text path, string
    :return: a dict with key = frameid and value is a list of lists [object id, x1, y1, x2, y2] in the frame, dict
    """
    # line format : [frameid, personid, x, y, w, h ...]
    with open(textpath) as f:
        f_csv = csv.reader(f)
        frames = {}
        for line in f_csv:
            if len(line) == 1:
                line = line[0].split(' ')
            # we only consider "pedestrian" class #
            if len(line) < 7 or (line[7] not in persons_class and "MOT2015" not in textpath) or int(float(line[6]))==0:
                continue
            if not (line[0]) in frames:
                frames[line[0]] = []
            bbox = xywh2xyxy(line[2:6])
            frames[line[0]].append([line[1]]+bbox)
    ordered = reorder_frameID(frames)
    return ordered

def calculate_iou(bbox,gt_box):
    x0,y0,x1,y1=int(bbox[1]),int(bbox[2]),int(bbox[3]),int(bbox[4])
    x0_b,y0_b,x1_b,y1_b=int(gt_box[1]),int(gt_box[2]),int(gt_box[3]),int(gt_box[4])
    area_a,area_b=(x1-x0)*(y1-y0),(x1_b-x0_b)*(y1_b-y0_b)
    iou_x0=np.maximum(x0,x0_b)
    iou_x1=np.minimum(x1,x1_b)
    iou_y0=np.maximum(y0,y0_b)
    iou_y1=np.minimum(y1,y1_b)

    iou_w=iou_x1-iou_x0
    iou_h=iou_y1-iou_y0
    area_iou=iou_w*iou_h
    iou=area_iou/ (area_a + area_b - area_iou)

    return iou
