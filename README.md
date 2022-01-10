# MTCNN-Implementation
My implementation of MTCNN

### Prerequisites
* Numpy: `pip install numpy`
* CV2: `pip install opencv-python`
* Pytorch: Go [here](http://pytorch.org/) to learn how to install the correct version for your environment.  

### Usage
The provided code can be used to train and test a MTCNN model. Some pretrained weights are provided.
A trained model can be tested on a video stream (from a webcam) or on video files using the following commands:
`python realtime.py` 
`python from_file.py --file video.mp4`
The video will be fetched from files in the videos folder.
