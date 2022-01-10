# MTCNN-Implementation
My implementation of MTCNN.

### Prerequisites
The code was tested on python 3.8.
* Pytorch: Go [here](http://pytorch.org/) to learn how to install the correct version for your environment.  
* Pytorch-lightning: `pip install pytorch-lightning`
* CV2: `pip install opencv-python`

### Usage
The provided code can be used to train and test a MTCNN model. Some pretrained weights are provided.

A trained model can be tested on a video stream (from a webcam) following command:
```
python realtime.py
```
To test on a video file instead use this command:
```
python from_file.py --file video.mp4
```
The video will be fetched from files in the videos folder. 

<video src='videos/annotated/video.gif' width=180/>

### Prerained weights
The pretrained weights were computed by training the different subnets as described in the MTCNN paper. 
Thus PNet, RNet and ONet were trained on WIDER FACE to detect the position of the bounding box of faces.
ONet was also trained on facial landmark data found in the CelebA dataset.

### Reference
- [MTCNN: Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf)
- [WIDER FACE: A Face Detection Benchmark](http://shuoyang1213.me/WIDERFACE/)
- [CelebA: Large-scale CelebFaces Attributes Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
