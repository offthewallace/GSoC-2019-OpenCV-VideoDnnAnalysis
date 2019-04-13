import numpy as np
import argparse
import imutils
import time
import cv2
import os
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import numpy as np

from multiprocessing.pool import ThreadPool
from collections import deque
import pandas as pd 

class paraser(self,task,config):

    def __init__(self,task,config):


    def downloadFile():
        pass

    def preprocess():
        pass


class Video:
    def __init__(self, Videoconfigs):
        self.inputSize=Videoconfigs["inputSize"]
        self.writeVideo=Videoconfigs["writeVideo"]
        #self.threadn = 1
        self.threadn = cv.getNumberOfCPUs()
        self.pool = ThreadPool(processes = threadn)
        #self.pool = dests
        self.pending = deque()
        self.videoReader= cv2.VideoCapture(config['videoAddress'])

    def videoProcess(Model):
        #writer = None
        #(W, H) = (None, None)
        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                else cv2.CAP_PROP_FRAME_COUNT
            total = int(self.videoReader.get(prop))
            print("[INFO] {} total frames in video".format(total))

        # an error occurred while trying to determine the total
        # number of frames in the video file
        except:
            print("[INFO] could not determine # of frames in video")
            print("[INFO] no approx. completion time can be provided")
            total = -1
        output={}
        framesID=0
        last_frame_time = clock()
         while True:
            
            while len(pending) > 0 and pending[0].ready():
                frameDict=pending.popleft().get()
                output["framesID"]=frameDict
            if len(self.pending) < self.threadn: 
                (grabbed, frame) = self.videoReader.read()
                framesID+=1
                if not grabbed:
                    break
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.inputSize, self.inputSize),
                    swapRB=True, crop=False)
                t = clock()
                #frame_interval.update(t - last_frame_time)
                last_frame_time = t
                
                task = pool.apply_async(Model.process_frame, (blob.copy()))
                pending.append(task)
                #Model.process

        
        return output

    def videoToFrame(Videoconfigs):
        return 1

# parse the user configs 



class Model():
    def __init__(self, Task, modelConfigs):
        self.modelNet = None
        self.lastLayer=None
        #self.ln=None
        self.modelType = Task
        self.framework=modelConfigs['framework']
        self.labels = open(modelConfigs["modelLabels"]).read().strip().split("\n")
        self.weightAddress=modelConfigs["weightAddress"]
        self.configAddress=modelConfigs["configAddress"]
        self.blobSize=modelConfigs["inputSize"]
        #self.toDownload=config["toDownload"]
 
    def loadmodel(self):
        
        self.modelNet=cv2.dnn.readNet(self.configAddress, self.weightAddress,self.framework)
        layerNames = self.modelNet.getLayerNames()
        lastLayerId = self.modelNet.getLayerId(layerNames[-1])
        self.lastLayer = self.modelNet.getLayer(lastLayerId)
        #ln = self.modelNet.getLayerNames()
        


   
    def process(blob):
        self.modelNet.setInput(blob)
        self.modelNet.forward(outNames)

        #sepecific for the object detection model 
        output={}
        classIds = []
        confidences = []
        boxes = []
        if self.modelNet.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
            # Network produces output blob with a shape 1x1xNx7 where N is a number of
            # detections and an every detection is a vector of values
            # [batchId, classId, confidence, left, top, right, bottom]
            for out in outs:
                for detection in out[0, 0]:
                    confidence = detection[2]
                    if confidence > confThreshold:
                        left = int(detection[3])
                        top = int(detection[4])
                        right = int(detection[5])
                        bottom = int(detection[6])
                        width = right - left + 1
                        height = bottom - top + 1
                        classIds.append(int(detection[1]) - 1)  # Skip background label
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])
        elif self.lastLayer.type == 'DetectionOutput':
            # Network produces output blob with a shape 1x1xNx7 where N is a number of
            # detections and an every detection is a vector of values
            # [batchId, classId, confidence, left, top, right, bottom]
            for out in outs:
                for detection in out[0, 0]:
                    confidence = detection[2]
                    if confidence > confThreshold:
                        left = int(detection[3] * frameWidth)
                        top = int(detection[4] * frameHeight)
                        right = int(detection[5] * frameWidth)
                        bottom = int(detection[6] * frameHeight)
                        width = right - left + 1
                        height = bottom - top + 1
                        classIds.append(int(detection[1]) - 1)  # Skip background label
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])
        elif lastLayer.type == 'Region':
            # Network produces output blob with a shape NxC where N is a number of
            # detected objects and C is a number of classes + 4 where the first 4
            # numbers are [center_x, center_y, width, height]
            classIds = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > confThreshold:
                        center_x = int(detection[0] * frameWidth)
                        center_y = int(detection[1] * frameHeight)
                        width = int(detection[2] * frameWidth)
                        height = int(detection[3] * frameHeight)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        classIds.append(classId)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])
        else:
            print('Unknown output layer type: ' + lastLayer.type)
            exit()
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            tempDict={}
            tempDict['className']=classIds[i]
            tempDict['confidences']=confidences[i]
            tempDict['left']=left
            tempDict['top']=top
            tempDict['right']=left + width
            tempDict['bottom']=top + height

            output['box'+str(i)]=tempDict
        return output


def postprocess(outputDict):
    pass   

def videoAnalysis(videos,task,config):
    
    parser=parser(task,config)

    video=Video(videos,configs)

    model=Models(Task,config)
    model.loadmodel()

    outputDict = video.videoProcess(model)




    
    
    

   

        # initialize the video stream, pointer to output video file, and
        # frame dimensions
        #vs = cv2.VideoCapture(config["input"])
       
        #need to change to muti processing mode 


	        # if the frame was not grabbed, then we have reached the end
	        # of the stream
            if not grabbed:
                break

	        # if the frame dimensions are empty, grab them
	        if W is None or H is None:
		    (H, W) = frame.shape[:2]

	        # construct a blob from the input frame and then perform a forward
	        # pass of the YOLO object detector, giving us our bounding boxes
	        # and associated probabilities
	        

            # initialize our lists of detected bounding boxes, confidences,
            # and class IDs, respectively
            
        # release the file pointers
        print("[INFO] cleaning up...")
        writer.release()
        vs.release()



    else: return 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
	    help="path to input video")
    ap.add_argument("-o", "--output", required=True,
	    help="path to output video")
    ap.add_argument("-t", "--task", required=True,
	    help="base path to model directory")
    ap.add_argument("-m", "--model", required=True,
	    help="base path to model directory")
    ap.add_argument("-class", "--class", 
	    help="path to class file")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args())
    Result_File, WrittenVideo= cv2.video_DNN.videoAnalysis(videoes, Tasks, configs). videoAnalysis(args["video"],args["task"],args)


