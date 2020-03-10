import cv2 
from collections import defaultdict
from tqdm import tqdm
import numpy as np

class Tracking_Observer():
    def __init__(self,frame_list,size=(1920,1080)):
        self.color_list=[(0,0,0),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255)]
        self.size=size
        self.frame_list=frame_list
        #nested_dict = lambda: defaultdict(nested_dict)
        #self.Tracklet=nested_dict()
        self.Tracklet=dict()
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def append_track(self,i,n,bbox):
        self.max_frame=i
    def save_txt(self,i):
        result=[]
        for f in range(len(self.Tracklet)):
            for j in range(len(self.Tracklet[f+1])):
                bbox=self.Tracklet[f+1][j]
                result.append([str(f+1)]+[bbox[0]]+[bbox[1],bbox[2],bbox[3]-bbox[1],bbox[4]-bbox[2]]+[-1,-1,-1,-1])
        np.savetxt("result.txt", np.array(result).astype(int), fmt='%i')

    def delete_id(self,i,n):
        print(self.Tracklet)
        self.Tracklet[i].pop(n)
        print(self.Tracklet[i])

    
    def make_video(self,path="track_result.mp4"):
        print("make_video")
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        video = cv2.VideoWriter(path, fourcc, 20.0, self.size)
        for f in tqdm(range(len(self.Tracklet))):
            frame=cv2.imread(self.frame_list[f])
            for j in range(len(self.Tracklet[f+1])):
                bbox=self.Tracklet[f+1][j]

                n,x1,y1,x2,y2=int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),int(bbox[4])
                cv2.rectangle(frame,(x1,y1),(x2,y2),self.color_list[n%8],5)
                cv2.putText(frame,'{}'.format(n),(x1,y1), self.font, 2,self.color_list[n%8],2,cv2.LINE_AA)
            cv2.imwrite("a.png",frame)
            video.write(frame)
        video.release()
        print("Done!!")




