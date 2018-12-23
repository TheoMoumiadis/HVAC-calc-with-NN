# HVAC-calc-with-NN
Deep Learning python code for HVAC cooling / Heating load calculation

Project Scope:
The project examines how a neural network can be applied within a design task of HVAC design, I decided to model a very common and fundamental process.

‘The initial calculation of cooling and heating loads for a medium size building’.

So the task became:

How to create a tool (trained AI model), which can predict the cooling and heating load of a medium size building by just providing some inputs without any engineering calculations.

Dataset used:
The dataset used was an existing collection from UCI Machine Learning repository [1] under the ownership and license of Athanasios Tsanas and Angeliki Xifara [2]. 

Model:
The problem was modeled through a 3-layer neural network algorithm including 2 hidden layers, 64 nodes per hidden layer and 0.01 as the regularization parameter. The input layer contains 8 normalized input parameters, and the output layer 1 variable


References
[1] https://archive.ics.uci.edu/ml/datasets/Energy+efficiency
[2] A. Tsanas, A. Xifara: "Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools", Energy and Buildings, Vol. 49, pp. 560-567, 2012
