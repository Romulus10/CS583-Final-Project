"""
Adapted from "The Implementation of Optical Flow in Neural Networks", Nicole Ku'ulei-lani Flett
"""

import numpy as np
from keras.models import model_from_yaml
import cv2


def loadNN(model_file,weights_file):
    yaml_file = open(model_file, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    loaded_model.load_weights(weights_file)
    print("Loaded model from disk")
    return loaded_model


def LK(frame1, frame2, model):
    img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    nn_input = np.concatenate((img1, img2), axis=0)
    nn_input.resize((28, 28))
    nn_input = np.reshape(nn_input, (1, 28, 28, -1))
    nn_output = model.predict(nn_input)
    print nn_output
    LKvector = [nn_output[0][1]-nn_output[0][0], nn_output[0][3]- nn_output[0][2], nn_output[0][5]-nn_output[0][4]]
    print LKvector
    return LKvector


recording = []
mod = loadNN('model.yaml','model.h5')
cap = cv2.VideoCapture(1)
ret1, frame1 = cap.read()
ret2, frame2 = cap.read()
while(ret1):
    frame1, frame2 = frame2, frame1
    ret, frame2 = cap.read()
    recording.append(LK(frame1, frame2, mod))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()