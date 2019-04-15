import numpy as np
import argparse
import imutils
import time
import cv2
import os
from __future__ import print_function

#from abc import ABCMeta, abstractmethod
import numpy as np

from multiprocessing.pool import ThreadPool
from collections import deque
import pandas as pd 

tasks=['object detection','classcification']
MODELADDRESS='list_topologies.yml'

class paraser:

    def __init__(self,task,config):
        self.videoAddress=config["video"]
        self.task=task
        self.confidence=config['confidence']
        self.threshold=config['threshold']
        self.classlabel=config['class']
        self.modelinfo=None
        self.modelWeight=None
        self.backend=config['backend']
        self.write=config['write']
        self.multiThread=config['multiThread']
        self.chooseModel=None
        self.inputSize=None
    
        def preprocess(self,config):
            if self.task not in tasks:
                print('Unknown task: ' + self.task)
                exit()
            if not config["modelChoose"]:
                #default model 
                if self.task is "object detection":
                    self.chooseModel="mobileNet-SSD"
                if self.task is "classcification":
                    self.chooseModel="mobileNet"
            else: self.chooseModel=config['modelChoose']
            if not config["model"]:
                self.modelWeight,self.modelConfigs=downloadModel(self,MODELADDRESS)
            else:
                self.modelWeight,self.modelinfo=config["modelWeight"], config['modelinfo']
            if not config["inputSize"]:
                self.inputSize=findInputsize(self.modelinfo)

            modelConfigs={'framework':self.backend,
                            "modelTask":self.task,
                            "modelLabels":self.classlabel
                            "weightAddress":self.modelWeight
                            "configAddress":self.modelinfo
                            "inputSize":self.inputSize}

            Videoconfigs={"inputSize": self.inputSize,
                            "writeVideo": self.,
                            'videoAddress': }

        def downloadModel(self,YAML):
            pass

        def findInputSize(self.modelinfo):
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
    '''
    def videoToFrame(Videoconfigs):
        return 1
    '''
# parse the user configs 



class Model():
    def __init__(self, modelConfigs):
        self.modelNet = None
        self.lastLayer=None
        #self.ln=None
        self.modelTask = Task
        self.framework=modelConfigs['framework']
        self.labels = open(modelConfigs["modelLabels"]).read().strip().split("\n")
        self.weightAddress=modelConfigs["weightAddress"]
        self.configAddress=modelConfigs["configAddress"]
        #self.blobSize=modelConfigs["inputSize"]
        #self.toDownload=config["toDownload"]
 
    def loadmodel(self):
        
        self.modelNet=cv2.dnn.readNet(self.configAddress, self.weightAddress,self.framework)
        layerNames = self.modelNet.getLayerNames()
        lastLayerId = self.modelNet.getLayerId(layerNames[-1])
        self.lastLayer = self.modelNet.getLayer(lastLayerId)
        #ln = self.modelNet.getLayerNames()
        

    ### to check the method has to be 
   def process(self,blob):
        if self.modelTask="object detection":
            if self.framework="darknet":
                return self.process_yolo(blob)
            else:
                return self.process_od(blob)
        else:
            return self.process_classcification(blob)
            
    def process_yolo(self,blob,frameWidth,frameHeight):
        #net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        output={}
        ln = net.getLayerNames()
        self.lastLayer = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        self.modelNet.setInput(blob)
        layerOutputs = net.forward(ln)
        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
        # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([frameWidth,frameHeight,frameWidth,frameHeight])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence,
        self.threshold)
        if len(idxs) > 0:
        # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                x, y,w,h = boxes[i][0], boxes[i][1],boxes[i][2], boxes[i][3]
                tempDict={}
                tempDict['className']=classIds[i]
                tempDict['confidences']=confidences[i]
                tempDict['left']=x
                tempDict['top']=y
                tempDict['right']=x + w
                tempDict['bottom']=y + h
                output['box'+str(i)]=tempDict
        return output

    
    def process_classcification(self,blob):
        pass

    def process_od(self,blob,frameWidth,frameHeight):
        self.modelNet.setInput(blob)
        outs=self.modelNet.forward(outNames)
        
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
                    if confidence > self.confThreshold:
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
                    if confidence > self.confThreshold:
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
                    if confidence > self.confThreshold:
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
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
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

def videoAnalyze(videos,task,config):
    
    parser=parser(task,config)

    video=Video(videos,configs)

    model=Models(Task,config)
    model.loadmodel()

    outputDict = video.videoProcess(model)
    return outputDict

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
	    help="path to input video")
    ap.add_argument("-o", "--output", default="./",
	    help="path to output video")
    ap.add_argument("-t", "--task", required=True,
	    help="base path to model directory")
    ap.add_argument("-b", "--backend", default='caffe',
        help="modelbackend ")
    ap.add_argument("-choose", "--modelChoose", 
        help="which model to use like ssd or FRCNN ")
    ap.add_argument("-m", "--model", 
	    help="base path to model directory")
    ap.add_argument("-info", "--modelinfo", 
        help="base path to modelinfo file address for tensorflow and caffe")
    ap.add_argument("-class", "--class",default="coco.name", 
	    help="path to class file")
    ap.add_argument("-multiThread", "--multiThread",type=bool,default="True", 
        help="use multiThread or not ")
    ap.add_argument("-write", "--write",type=bool,default="True", 
        help="write the result on video or not ")

    ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args())

    Result_File=videoAnalyze(args["video"],args["task"],args)


