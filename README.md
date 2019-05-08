# CS542 SmartGrasp
Spring 2019 CS542 machine learning course project at Boston University

## Preprocessing the dataset

Before you start, you need to get the dataset from https://www.kaggle.com/ugocupcic/grasping-dataset.

### Split a small chunk from the original dataset

* splitData.py
  
We have a little Python script in this repository which can be used to split a chunk of data from the original dataset. It is called 'splitData.py'. The number of datapoints(feature, label) is controlled by the number of iterations on line 3.

This script will generate a .mat file which can be read by MATLAB. Further processing includes extracting suitable features and labels for training and testing is done in MATLAB. Please see the next section.

### Generate training data

* preprocess.m

We use a script in MATLAB to extract the label and features from raw data. To run this script, please put the small chunk of raw data generate in the previous step into your MATLAB search path and modify the name of the dataset on line 3 and line 4 to match the name of the smaller chunk. 

## Training prediction model

We use two different classification methods in this project. Please make sure you have ***scikit-learn*** and ***Keras*** installed on your machine.

* svm.py

You could run this script to train a SVM prediction model. Before you run the script, please make sure you have preprocessed dataset. The file name of dataset could be changed in line 46. Also, you could change the kernel function and other parameters of SVM in line 27.

* nn.py

You could run this script to train a neural network. You could change the parameter of neural network in line 72.

## Test the prediction model in Robot Simulator

introduction: This test code is developed based on the docker environment from Shadow Robot Company. We adapt the environment to be runnable with latest version of Keras and put our neural network model in it. Test code using ROS for our own model is developed based on previous work found in the docker image.

prerequisite: please make sure you have docker installed on your machine.

steps to run the test:

1. pull and run the docker image from docker hub.

    docker run -it --name sgs -p 8080:8080 -p 8888:8888 -p 8181:8181 -p 7681:7681 yxiao1996/cs542smartgrasp

2. check the jupyter notebook client at localhost:8888 and Gazebo simulator at localhost:8080 
3. run the grasp command in jupyter notebook.
4. run the prediction note book in jupyter notebook.
