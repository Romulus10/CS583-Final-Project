# CS583-Final-Project

Optical Flow in Neural Networks


Abstract:

Optical flow is a method of estimation of the relative positions or apparent motion of objects in
subsequent frames. There are a few methods of estimation of optical flow. The paper cited here uses the
Lucas-Kanade method and convolutional neural networks to calculate optical flow, with the goal of
introducing a procedure for fine-tuning of convolutional neural networks for optical flow estimation.
There are a few areas of improvement, enumerated below, in the cited paper that we hope to address in
our project. In this work, we shall demonstrate improvements to these issues with the above paper’s
standing process for processing optical flow with a convolutional neural network by employing the
pyramidal approach to the Lucas-Kanade method. We shall implement mini-batch gradient descent over
the established stochastic gradient descent to further optimize the learning function employed by the
paper and determine the process’ capacity to be trained by an animated scenario, rather than purely by
real-life footage. We shall finally test the performance and behavior benefits of an eigenvalue-based
approach to the Lucas-Kanade method of optical flow tracking over the least square method.
