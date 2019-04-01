from math import sqrt
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  1/(1+np.exp(-z))


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    label_matrix = []
    for label in training_label:
        label_matrix.append([0 if x != label else 1 for x in range(2)])
    label_matrix = np.array(label_matrix)

    #
    # Feedforward Propagation
    input_bias = np.ones((training_data.shape[0],1)) # create an bias
    training_data_bias = np.concatenate((training_data, input_bias), axis=1) # add bias to training data
    hiden_out = sigmoid(np.dot(training_data_bias, w1.T))  # 3.32 equtions 1 and 2
    hiden_bias = np.ones((1,hiden_out.T.shape[1])) # create an bias
    hiden_out_bias = np.concatenate((hiden_out.T, hiden_bias), axis=0).T  # add bias to hiden_out data
    net_out = sigmoid(np.dot(hiden_out_bias,w2.T)) # 3.32 eqution 3 and 4, feed forward is complete.

    # comupute the obj_val
    first_term1 = np.dot((label_matrix).flatten(),(np.log(net_out)).flatten().T)
    #first_term2 = np.dot((np.ones(label_matrix.shape)-label_matrix).flatten(),(np.log(np.ones(net_out.shape)-net_out).flatten().T))
    first_term2 = np.dot(np.array(1-label_matrix).flatten(),np.log((np.array(1-net_out)).flatten().T))
    #first_term = np.dot((label_matrix).flatten(),(np.log(net_out)).flatten().T)+np.dot(np.array(1-label_matrix).flatten(),(np.log(np.array(1-net_out)).flatten().T)
    #s
    first_term = first_term1+first_term2
    second_term = lambdaval*(np.dot(w1.flatten(),w1.flatten().T)+np.dot(w2.flatten(),w2.flatten().T))
    obj_val = -1/(2*training_data.shape[1])*first_term + second_term / (2*training_data.shape[0]) # finish off eqn (15)



    # Error function and Backpropagation
    #delta_l = np.array(net_out)*np.array(1-net_out)*np.array(label_matrix - net_out) # correspondes to eqn(9)
    delta_l = net_out-label_matrix
    #dev_lj = -1*np.dot(delta_l.T, hiden_out_bias) # correspondes to eqn(8)
    dev_lj = np.dot(delta_l.T, hiden_out_bias)
    grad_w2 = (dev_lj + lambdaval *w2)/ training_data.shape[0] #correspondes to eqn(16)
    w2_noBias = w2[:,0:-1]
    delta_j = np.array(hiden_out)*np.array(1-hiden_out) # correspondes to -(1-Zj)Zj in eqn(12)
    dev_ji = np.dot((np.array(delta_j)*np.array(np.dot(delta_l,w2_noBias))).T,training_data_bias) # correspondes to eqn(12)
    grad_w1 = (dev_ji+lambdaval*w1)/training_data.shape[0] #correnspondes to eqn(17)


    # Reshape the gradient matrices to a 1D array.
    grad_w1_reshape = np.ndarray.flatten(grad_w1.reshape((grad_w1.shape[0]*grad_w1.shape[1],1)))
    grad_w2_reshape = grad_w2.flatten()
    obj_grad_temp = np.concatenate((grad_w1_reshape.flatten(), grad_w2_reshape.flatten()),0)
    obj_grad = np.ndarray.flatten(obj_grad_temp)



    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    return (obj_val, obj_grad)

n_input = 5
n_hidden = 3
n_class = 2
training_data = np.array([np.linspace(0,1,num=5),np.linspace(1,0,num=5)])
training_label = np.array([0,1])
lambdaval = 0
params = np.linspace(-5,5, num=26)
args = (n_input, n_hidden, n_class, training_data, training_label, lambdaval)
objval,objgrad = nnObjFunction(params, *args)
print(objval)
print(objgrad)
