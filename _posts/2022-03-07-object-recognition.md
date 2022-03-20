
---
title: "Intro to object recognition and convnets"
description: "Part [1/3] of my Road to Masters story"
layout: post
toc: true
branch: master
badges: false
image: images/ipynb/rtm_object_recognition.png
comments: false
author: Giaco Stantino
categories: [computer vision, machine learning]
hide: false
search_exclude: true
permalink: /blog/object-recognition

---

# <center>Intro</center>

Object recognition is a computer vision technique for identifying objects in images or videos. Object recognition is a key technology behind driverless cars, enabling them to recognize a stop sign or to distinguish a pedestrian from a lamppost. It is also useful in a variety of applications such as disease identification in bioimaging, industrial inspection, and robotic vision.

<center><img src="https://giacostantino.com/images/ipynb/rtm_object_recognition.png"></center>

This blog post is a result of my deep interest in neural networks. It is also the first part of three epsiode series, which sums up my experiences with computer vision until my master's thesis, which is the final part. All of the posts are discussing the core concepts and ideas behind the projects. However for this part I uploaded the notebook to [my github repo](#).

# Neural Networks
This is one of the prerequists for this blogpost. It would be best if you understand the concept - here is [a good article](https://www.ibm.com/cloud/learn/neural-networks).<br>
tldr: <br>
Neural Networks are algorithms using an artificial neurons. They compute output value based on inputs *(x1, x2, x3)* - to do that they learn input weights and then pass it to activation function.

<center><img src="https://giacostantino.com/images/ipynb/convnet_neuron.png"></center>

For indepth info check: [Michael Nielsen's internet book](http://neuralnetworksanddeeplearning.com/index.html)

# Deep Neural Networks
Neural Network that use multiple layers of artifical neurons, are called deep neural networks.

<center><img src="https://giacostantino.com/images/ipynb/convnet_deepnn.png" width=570></center>

Above we can see the most basic concept of MLP (multi layer perceptron). Such models are characterized by a hierarchical structure, i.e. transformations take place on many layers, thanks to which you can take advantage of the possibility of returning. At each layer created is a data abstraction representation, another one for a different process for equated neural networks that are inherited.

The use of deep neural networks for image analysis is a promising idea. We assign each pixel in the grayscale image a numeric value, such as 0 for black to 1 for white. For a 16px x 16px image, there are 256 values ​​that are passed as input values. In the case of the color scale, the colors of the image are broken down into three RGB channels, which gives 3x256 values ​​of the input data. However, the transmission of information from pixels directly to the MLP neurons results in the appearance of the following problem

<center><img src="https://giacostantino.com/images/ipynb/convnet_cat.png"></center>

The figure above shows the frame of the detected cat's head. The deep neural network in the learning process will only acquire the face recognition insights at this image location - the corresponding weights will be modified - as shown on the right. In other words, the network will learn to recognize a cat in specific locations. In order to solve this problem, the architecture of convolutional networks was proposed.

# ConvNets

These networks get their name from the mathematical convolution operation. The convolutional neural network consists of an input layer, convolutional layers and an output layer. On [CNN](https://sgugger.github.io/convolution-in-depth.html), hidden layers perform convolutions. 

<center><img src="https://giacostantino.com/images/ipynb/convnet_convnn.png" width=570></center>

As shown above,  a convolutional layer is one that takes the dot product of the [kernel](https://setosa.io/ev/image-kernels/) with the input matrix.  The convolution operation generates a feature map, which in turn affects the input for the next layer. This is followed by other layers such as pool layers, or fully linked layers.

<center><img src="https://giacostantino.com/images/ipynb/convnet_convnn_layer.png" ></center>

**The concept of convolutional layers** is based on simple assumptions:
1. nearby pixels are more important than far apart pixels,
2. objects are made up of smaller parts.

The influence of the surrounding pixels is analyzed using filters, usually 3x3 or 5x5. Such a filter moves across the entire image, the step of this shift is called a stride. Above figure shows a simple convolutional operation of applying a filter to the input pixel. For each point in the image, an output value is computed using convolution operations. The matrix of those outputs is called **feature map**.

The pooling layers then extract the most important information from the feature maps, get rid of the noise, and thus condense the information into smaller maps. Finally, the feature map neurons are flattened into a vector form and fully connected to the next layer that generates predictions.

Although Convolution Networks produce acceptable outputs there was one more evolution to be made (well, a few more in fact, but we will consider today this one).

# ResNets

 One of the problems of  the neural network is that with each successive layer it becomes more difficult to propagate information from the early layers and the accuracy of the network results decreases - this is known as the **degradation problem**. As a solution to this problem, residual blocks, which are a modification of convolutional blocks, have been proposed

<center><img src="https://giacostantino.com/images/ipynb/convnet_convnn_residual.png"></center>

Let F (x) be a function learned by the layers. Consider the output y = F (x) + x, for a block omitting connections. Here the term + x means a combination of the so-called "residual". For y = F (x) + x, where the +x component carries the original value, the block only needs to learn the value changes, i.e. the residual Δx. In other words, this approach **changes the 'intuition'** of training from simply recognizing features to learning the differences between them. Hence the name "Residual Learning".

# Object Recognition

Object recognition is a computer vision technique used to identify objects in images or videos. Object recognition is a key result of deep learning and machine learning algorithms. When people look at a photo or watch a video, we can easily see people, objects, scenes, and visual details. The goal is to teach the computer to do what naturally comes to a person: gain a level of understanding of what the picture contains.

<center><img src="https://giacostantino.com/images/ipynb/convnet_detection.png"></center>

In the project in my repository I trained a resnet model to classify animals base on their picture. Here are some sample results.

<center><img src="https://giacostantino.com/images/ipynb/convnet_piesek.png" ></center>

Model's performance in evaluated based on it's accuracy score. In perfect world it would be so easy to assess if our experiments end with a satisifing result. Unfortunately, we don't live on ideal planet. One of the problems of the neural networks that they might overfit during the training. In other words, they will learn by heart without understandind the essensce of the data. If put to the tests, they will score best on pictures their learned with, but perform miserably with data they see first time.

# Overfitting
To prevent model from learing by heart (overfitting) we need to spot it happening first. The very basic tool for this is *learning curve*. There are ploted results of the model on training data and test (unseen) data by number of epochs (number of times model seen whole dataset).

<center><img src="https://giacostantino.com/images/ipynb/convnet_learning_curve.png"></center>

Above I plotted the learning curve for my animals recognition project. It turns out that model starts overfiting after epoch 14 - when validation and traning curve drift apart and the accuracy graph stops growing. 

To prevent this from happening earlier, my model uses a few *regularization* techniques:

 - [scheduled learning rate](https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1)
 - [gradient clipping](https://machinelearningmastery.com/how-to-avoid-exploding-gradients-in-neural-networks-with-gradient-clipping/)
 - [weight decay](https://programmathically.com/weight-decay-in-neural-networks/)

> :note: There is a lot of diffrent techniques. I chose the ones that, in my opinion, suited the project best.

# Summary

The neural network model is a powerfull tool and is best suited for tasks where traditional machine learning algorithms underperform, such as computer vision and speech recognition. In the post we discussed the core ideas behind the object recognition, yet there is a lot lef unsaid: neuron activation function, loss function, optimization method... Some of them I am going to discuss in upcoming post considering object detction for colision signaling. Stay tuned. 
