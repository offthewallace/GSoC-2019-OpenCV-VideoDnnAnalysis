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




class Video:
    def __init__(self, videoAddress,configs):
        if configs['mutiThread']:
            self.applyMutiThread=True
        else: self.applyMutiThread=False
        self.writeVideo=False
        self.threadn = 1
        self.pool = dests
        self.pending = deque()
        self.videoReader= cv2.VideoCapture(videoAddress)
    def videoProcess(model):
        writer = None
        (W, H) = (None, None)
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
         while True:
            while len(pending) > 0 and pending[0].ready():
                (grabbed, frame) = vs.read()
    def videoToFrame(configs):
        return 1

# parse the user configs 
class paraser


class Model():
    def __init__(self, Task, configs):
        self.modelNet = None
        #self.ln=None
        self.modelType = Task
        self.framework=config['framework']
        self.labels = open(config["modelLabels"]).read().strip().split("\n")
        self.weightAddress=config["weightAddress"]
        self.configAddress=config["configAddress"]
        self.blobSize=config["inputSize"]
        #self.toDownload=config["toDownload"]


    '''
    def load():
        pass
    def process():
        pass


    class loader():

    class processor():

    def downloadModel(self):
        #should that shit put in paraser class ? so we only focus on load

        return 1

    def modelImport()

    '''
    #control by lastLayer type 
    def loadmodel(self):
        
        self.modelNet=cv2.dnn.readNet(self.configAddress, self.weightAddress,self.framework)
        layerNames = self.modelNet.getLayerNames()
        lastLayerId = self.modelNet.getLayerId(layerNames[-1])
        lastLayer = self.modelNet.getLayer(lastLayerId)
        #ln = self.modelNet.getLayerNames()
        return lastLayer


    def process_helper(blob):
        self.net.setInput(blob)
        self.net.forward(outNames)
       
    # check the decorator Its wrong in Here!!!
    @process_helper(blob)
    def process(out,lastLayer):
        classIds = []
        confidences = []
        boxes = []
        if self.net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
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
        elif lastLayer.type == 'DetectionOutput':
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
        


    def process_frame_object_detection(self,blob):
        COLORS=  np.random.randint(0, 255, size=(len(self.labels), 3),dtype="uint8")
        
        self.modelNet.setInput(blob)
        start = time.time()
        layerOutputs = self.modelNet.forward(self.ln)
        end = time.time()
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
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
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


            # apply non-maxima suppression to suppress weak, overlapping
            # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence,self.threshold)

        # ensure at least one detection exists
        output = {}
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]

                # draw a bounding box rectangle and label on the frame
                if self.write==True:
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(self.labels[classIDs[i]],
                        confidences[i])
                    cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                output.append([])
        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(self."output", fourcc, 30,
                (frame.shape[1], frame.shape[0]), True)
            writer.write(frame)
            # some information on processing single frame
            #if total > 0:
            #    elap = (end - start)
            #    print("[INFO] single frame took {:.4f} seconds".format(elap))
            #    print("[INFO] estimated total time to finish: {:.4f}".format(
            #        elap * total))

        # write the output frame to disk
        return 

    # some intensive computation...
    
        
        

def videoAnalysis(videos,task,config):
    
    video=Video(videos,configs)
    model=Models(Task,config)
    
    
    threadn = cv.getNumberOfCPUs()
    pool = ThreadPool(processes = threadn)
    

    if task=="yolo":
        weightsPath = os.path.sep.join([config["model"], task+".weights"])
        configPath = os.path.sep.join([config["model"], task+".cfg"])
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

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


