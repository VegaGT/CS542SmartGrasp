# CS542SmartGrasp
Spring 2019 CS542 machine learning course project at Boston University


## Test the prediction model in Robot Simulator

introduction: This test code is developed based on the docker environment from Shadow Robot Company. We adapt the environment to be runnable with latest version of Keras and put our neural network model in it. Test code using ROS for our own model is developed based on previous work found in the docker image.

prerequisite: please make sure you have docker installed on your machine.

steps to run the test:

1. pull and run the docker image from docker hub.

    docker run -it --name sgs -p 8080:8080 -p 8888:8888 -p 8181:8181 -p 7681:7681 yxiao1996/cs542smartgrasp

2. check the jupyter notebook client at localhost:8888 and Gazebo simulator at localhost:8080 
3. run the grasp command in jupyter notebook.
4. run the prediction note book in jupyter notebook.