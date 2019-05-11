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

You could run this script to train a neural network. You could change the parameter of neural network in line 72. The model is saved as json file, and the weights of neural network is saved as h5 file.

* experiment.m

The experiment.m is a MATLAB script we use at the beginning of the project. We use this script to test our thoughts. It is not a part of our final results.

## Test the prediction model in Robot Simulator

Introduction: This test code is developed based on the docker environment from Shadow Robot Company. We adapt the environment to be runnable with latest version of Keras and put our neural network model in it. Test code using ROS for our own model is developed based on previous work found in the docker image.

Prerequisite: please make sure you have docker installed on your machine.

Steps to run the test:

0. make sure you have docker installed on your machine.
1. pull and run the docker image from docker hub.

    docker run -it --name sgs -p 8080:8080 -p 8888:8888 -p 8181:8181 -p 7681:7681 yxiao1996/cs542smartgrasp

2. check the jupyter notebook client at localhost:8888 and Gazebo simulator at localhost:8080. you should be able to see a simulated environment with a table, a red ball and a robotic arm in the simulator. 
3. run the grasp command in jupyter notebook. please find the "Smart Grasping Sandbox.ipynb". by running the fist three blocks, you should be able to observe that the robotic hand would move towards the red ball and try to grasp it.
4. run the prediction notebook in jupyter notebook. please find the "DisplayGraspQualiyuNeuralNet.ipynb", the displaying block is the final block in this notebook. to reproduce the result, please first send command to the robotic arm for grasping in "Smart Grasping Sandbox.ipynb" and then run all blocks in "DisplayGraspQualiyuNeuralNet.ipynb". you should be able to observe the jump of prediction as we shown in presentation. 
