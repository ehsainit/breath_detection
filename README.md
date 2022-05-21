# Breath Detector

Real-time contactless respiratory rate detector based on combination of image processing and DL techniques


## Motivation

The recent pandemic helped to realize that humans relay on contact-based technologies
in many applications. Hence, as a reaction a new trend of contactless technologies is being
invented across various domains specially in the medical sector.

## Proposal

Here we propose a robust method based on contactless technologies to assess the respiratory rate of 
a stationary human being in real-time. This project will incorporate state-of-the-art object detection 
neural network called YOLOv4 to efficiently detected a human face and use a low-cost thermo camera 
to measure the breathing rate based on difference in human face heat map during inhalation and exhalation.
Moreover, a low-cost depth camera will be utilized to improve the performance by measure the change
in human chest/belly z-distance relative to the camera.

### YOLOv4 Face Detector

For the DL part one of the many existing face detectors weight files will be utilized
Here is a list of the reviewed once so far (will be updated):
 - https://github.com/akshat235/face-detection-yolov4-tiny