# GSoC-2019-OpenCV-VideoDnnAnalysis
Its for demo of GSoC-2019-OpenCV-VideoDnnAnalysis proposal 

## Overview

This project is a highlevel API based on OpenCV library that can help users to more easily
process different deep learning models on videos.

the goal would be for users to ultimately only have to call one-line helper methods that return the processing results in
JSON/DataFrame format for future data analysis.

## development log:
Update 4/14: support YOLO now. I will add more features and better class design and documentation like in 2 hours. 


Update 4/14: adding download file and parser for the class design, need finish download and findinputsize today and adding documentation. 


Update 4/13: adding supporting the basic object detection muti-tasks ( FRCNN/SSD/YOLO) and Muti-threading(for python it would be multi processing because of GLI lock)


Update 4/13: looks like the read model would from standard function readframe but the network's output would be different. So the thing i need to do is to create a dictionary or config file for the network's output, along with the standard label and config. I will change it from the Yaml file from the OpenCV Zoo.


Update 4/12: change to the abstract class model, will be done around 6 pm EST. I will try the Decorator pattern instead of the abs factory 


