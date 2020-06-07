# Optical Flow in Neural Networks

## *OpitcalFlow.pdf*
describes our work in detail. Please refer to the paper as a starter.

## *OpticalFlow.ipynb*
is the Jupyter notebook where all the action is. It walks through the process of the calculation of optical flow of the training fata using the pyramidal Lucas-Kanade approach, training the CNN model using the results from the LK method, and testing the model's results with test data.

## Description of other files

### *optical_flow.py*
Contains code that calculates optical flow using the pyramidal Lucas-Kanade to be used to train the CNN

### *Lkvf.py*
Contains code related to training the CNN model

### *RunLKNN.py*
Contains code to calculate optical flow using the CNN
