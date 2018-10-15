# VideoCaptioning
## Introduction

This is the baseline model for video captioning with Pytorch. Recent work with more innovative methods can not be public for the time being.

This is the novice practice project and the code was written when I just became a postgraduate student. Due to the urgency of time, the collation of the relatively hasty. I will continue to add comments and so on in the following time.

If you have any questions, please let me know. 

## Dependencies

All code is writen in python. To use it you will need:

* Python
* Numpy
* Scipy
* Pytorch
* h5py
* pickle
* PIL

## Usage

### Take features

In order to save time in traing, you need to take frame features of videos at first. Therefore, in the course of traing does not need to deal with the original images and don't need to put them into the convolutional network.

I provide to python file for taking features of frames. Actually, most of the code for the files are the same. I use different files because of the input format of training and testing is different.

```
python take_feature.py/take_test_feature.py
```

### For training

The model's input is the feature has been taken with pre-trained CNN which can improve the speed of training. The final scores are the same with putting original images.

```
python train_video_features.py
```

### For testing

Generate captions for the val/test split of dataset and output the final scores (include Bleu/Cider/Meteor/Rougel).

```
python test_features.py or score.py
```
