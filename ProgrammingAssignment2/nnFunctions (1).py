import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import gradient
from scipy.optimize import minimize
from math import sqrt
import pickle
'''
You need to modify the functions except for initializeWeights() 
'''

def initializeWeights(n_in, n_out):
    '''
    initializeWeights return the random weights for Neural Network given the
    number of node in the input layer and output layer

    Input:
    n_in: number of nodes of the input layer
    n_out: number of nodes of the output layer

    Output:
    W: matrix of random initial weights with size (n_out x (n_in + 1))
    '''
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    '''
    Notice that z can be a scalar, a vector or a matrix
    return the sigmoid of input z (same dimensions as z)
    '''
    # remove the next line and replace it with your code
    z = 1.0 / (1.0 + np.exp(-z))
    return z 

def nnObjFunction(params, *args):
    '''
    % nnObjFunction computes the value of objective function (cross-entropy
    % with regularization) given the weights and the training data and lambda
    % - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices W1 (weights of connections from
    %     input layer to hidden layer) and W2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of nodes in input layer (not including the bias node)
    % n_hidden: number of nodes in hidden layer (not including the bias node)
    % n_class: number of nodes in output layer (number of classes in
    %     classification problem
    % train_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of the corresponding instance 
    % train_label: the vector of true labels of training instances. Each entry
    %     in the vector represents the truee label of its corresponding training instance.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector (not a matrix) of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    '''
    # do not remove the next 5 lines
    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    # First reshape 'params' vector into 2 matrices of weights W1 and W2

    W1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # remove the next two lines and replace them with your code 
    obj_val = 0
    training_data_1 =  np.concatenate((train_data,np.ones((train_data.shape[0],1))),axis=1)
    w1_T_t1 = np.dot(training_data_1,np.transpose(W1))
    prediction_1 = sigmoid(w1_T_t1)
    training_data_2 = np.concatenate((prediction_1,np.ones((prediction_1.shape[0],1))),axis=1)
    w2_T_t2 = np.dot(training_data_2,np.transpose(W2))
    prediction_2 = sigmoid(w2_T_t2)
    y_predicted = np.zeros((train_data.shape[0],n_class))
    for i in range(0,train_label.shape[0]):
        y_predicted[i][(int)(train_label[i])]=1
    first_part_of_eq5 = np.multiply(y_predicted,np.log(prediction_2))
    second_part_of_eq5 = np.multiply(np.subtract(1,y_predicted),np.log(np.subtract(1,prediction_2)))

    eq5 = np.divide(np.sum(np.add(first_part_of_eq5,second_part_of_eq5)),(-1*train_data.shape[0]))
    regularized_term = (lambdaval/(2*train_data.shape[0]))*(np.add(np.sum(np.square(W1)),np.sum(np.square(W2))))
    obj_val = eq5 + regularized_term


    # #adding bias term to the train_data
    # bias1 = np.ones(train_data.shape[0],1)
    # train_data_1 = np.append(train_data,bias1,1)

    # #feed forwarding through 1st layer
    # W1_t = np.transpose(W1)
    # first_prediction = np.dot(train_data,W1_t)
    # sigmoid_of_first_prediction = sigmoid(first_prediction)

    # #adding bias term for further propogation
    # bias2 = np.ones(sigmoid_of_first_prediction.shape[0],1)
    # train_data_2 = np.append(sigmoid_of_first_prediction,bias2,1)

    # #feed forwarding through 1st layer
    # W2_t = np.transpose(W2)
    # second_prediction = np.dot(train_data_2,W2_t)
    # final_prediction = sigmoid(second_prediction)

    # #1-k encoding of train_label
    # y_predicted = np.zeros(train_data.shape[0],n_class)

    # i = 0
    # for i in range(train_label.shape[0]):y_predicted[i][train_label[i]]=1

    # obj_val = np.sum(np.sum(np.multiply(y_predicted,np.log(final_prediction)),np.multiply(np.subtract(1,y_predicted),np.log(np.subtract(1,final_prediction))))/(-1*train_data.shape[0]),np.multiply(lambdaval/(2*train_data.shape[0]),np.sum(np.sum(W1**2),np.sum(W2**2))))

    obj_grad = params
    lambda_l = np.subtract(prediction_2,y_predicted)

    gradient_2 = np.dot(np.transpose(lambda_l),training_data_2)
    regularized_gradient_2 = np.divide(np.add(gradient_2,np.multiply(lambdaval,W2)), train_data.shape[0])

    #training_data_2 = training_data_2[:,0:training_data_2.shape[1]-1]
    eq12 = np.multiply(np.multiply(np.subtract(1,prediction_1),prediction_1),lambda_l.dot(W2[:,0:W2.shape[1]-1]))
    gradient_1 = np.dot(np.transpose(eq12),training_data_1)
    #gradient_1 = gradient_1[:,0:gradient_1.shape[1]-1]
    regularized_gradient_1 = np.divide(np.add(gradient_1,np.multiply(lambdaval,W1)), train_data.shape[0])

    obj_grad = np.concatenate((regularized_gradient_1.flatten(),regularized_gradient_2.flatten()),0)

    # gradient_2 = np.divide(np.sum(np.dot(np.transpose(np.subtract(final_prediction,y_predicted)),train_data_2),np.multiply(lambdaval,W2)),train_data.shape[0])

    # gradient_1 = np.divide(np.sum(np.dot(np.transpose(np.dot(np.dot(np.subtract(final_prediction,y_predicted),W2),np.dot(train_data_2,np.subtract(1,train_data_2)))),train_data_1),np.multiply(lambdaval,W1)),train_data.shape[0])

    return (obj_val,obj_grad)

def nnPredict(W1, W2, data):
    '''
    % nnPredict predicts the label of data given the parameter W1, W2 of Neural
    % Network.

    % Input:
    % W1: matrix of weights for hidden layer units
    % W2: matrix of weights for output layer units
    % data: matrix of data. Each row of this matrix represents the feature vector for the corresponding data instance

    % Output:
    % label: a column vector of predicted labels
    '''
    # remove the next line and replace it with your code
    labels = np.zeros((data.shape[0],1))

    training_data_1 =  np.concatenate((data,np.ones((data.shape[0],1))),axis=1)
    w1_T_t1 = np.dot(training_data_1,np.transpose(W1))
    prediction_1 = sigmoid(w1_T_t1)
    training_data_2 = np.concatenate((prediction_1,np.ones((prediction_1.shape[0],1))),axis=1)
    w2_T_t2 = np.dot(training_data_2,np.transpose(W2))
    prediction_2 = sigmoid(w2_T_t2)

    labels = np.argmax(prediction_2,axis=1)

    return labels
