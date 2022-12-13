# COMP 576 Final Project Liveness Detection

## Rui Xiao, Yangfenghui Huang, Yusi Qi, Danfeng Yang

This is a project that combines OpenCV with deep learning to achieve real time facial liveness detection.

### Requirements
***
* Python
* OpenCV
* Tensorflow

### Usage
***

* To capture produce video for image capturing:

Use your webcam to record a 5 min video of yourself, this will be the real faces, then replay the video just taken to your webcam, this will be the fake faces.

* To produce the dataset:
```
python gather_examples.py --input videos/real.mov --output dataset/real --detector face_detector --skip 1

python gather_examples.py --input videos/fake.mov --output dataset/fake --detector face_detector --skip 4
```

* To train the model:
```
python train.py --dataset dataset --model liveness.model --le le.pickle

```
* To run a demo:
```
python liveness_demo.py --model liveness.model  --detector face_detector
```



# livenessnet

## train model

python train.py --dataset dataset --model liveness.model --le le.pickle

