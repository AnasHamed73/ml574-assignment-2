'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from math import sqrt
from math import exp
from scipy import optimize


# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W


# Replace this with your sigmoid implementation
def sigmoid(z):
    return 1/(1+exp(-z))  # your code here


# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    label_matrix = []
    for label in training_label:
        label_matrix.append([0 if x != label else 1 for x in range(10)])
    label_matrix = np.array(label_matrix)


    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    
    # Feedforward Propagation
    bias_1 = np.ones((training_data.shape[0],1))
    training_data_with_bias = np.concatenate((training_data, bias_1), axis=1)
    hidden_output = sigmoid(np.dot(training_data_with_bias,w1.T))
    bias_2 = np.ones((1,hidden_output.T.shape[1]))
    hidden_output_with_bias = np.concatenate((hidden_output.T,bias_2), axis=0).T
    Feedforward_output = sigmoid(np.dot(hidden_output_with_bias,w2.T))
    
    #obj_val
    n = training_data.shape[0]
    k = Feedforward_output.shape[1]
    
    
    ff = (np.log(Feedforward_output)).T
    gg = (np.log(1 - Feedforward_output)).T
    J_sum = np.array(-(np.dot(label_matrix, ff) + np.dot((1 - label_matrix), gg)))
    diagonal = np.zeros((np.shape(J_sum)[0], 1))
    for i in range(np.shape(J_sum)[0]):
        diagonal[i][0] = J_sum[i][i]
    J_w1_w2 = np.mean(diagonal, axis=0)

    regularization_term = np.dot(lambdaval,(np.dot(w1.flatten(),w1.flatten().T)+np.dot(w2.flatten(),w2.flatten().T)))
    obj_val = J_w1_w2.flatten() + regularization_term/np.dot(2,training_data.shape[0])

    #Backpropagation
    delta_l = np.array(label_matrix - Feedforward_output)
    derivative_lj = -1*np.dot(delta_l.T, hidden_output_with_bias)
    gradient_w2 = (derivative_lj + np.dot(lambdaval, w2))/training_data.shape[0]
    w2_without_bias = w2[:,0:-1]
    delta_j = np.array(hidden_output)*np.array(1-hidden_output)
    derivative_ji = -1*np.dot((np.array(delta_j)*np.array(np.dot(delta_l,w2_without_bias))).T,training_data_with_bias)
    gradient_w1 = (derivative_ji+np.dot(lambdaval,w1))/training_data.shape[0]
    
    #Reshape the gradient to a 1D array
    gradient_w1_reshape = np.ndarray.flatten(gradient_w1.reshape((gradient_w1.shape[0]*gradient_w1.shape[1],1)))
    gradient_w2_reshape = gradient_w2.flatten()
    obj_grad_temp = np.concatenate((gradient_w1_reshape.flatten(),gradient_w2_reshape.flatten()),0)
    obj_grad = np.ndarray.flatten(obj_grad_temp)
    
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])
    #print(obj_val, obj_grad)
    
    return (obj_val, obj_grad)


    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    labels = np.array([])
    # Your code here
    # print("Data shape: ", data.shape)
    # print("w1 shape: ", w1.shape)
    # print("w2 shape: ", w2.shape)
    
    ones_input = np.ones((np.shape(data)[0], 1))
    data = np.concatenate([ones_input, data], axis=1)
    a = sigmoid(np.dot(data, np.transpose(w1)))
    
    ones_hidden = np.ones((np.shape(a)[0], 1))
    a = np.concatenate([ones_hidden, a], axis=1)
    labels = sigmoid(np.dot(a, np.transpose(w2)))
    
    preds = np.zeros((np.shape(data)[0], 1))
    for i in range(np.shape(data)[0]):
        max = 0
        for j in range(np.shape(labels)[1]):
            if labels[i][j] > labels[i][max]:
                max = j
        preds[i] = max
    
    return preds


# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter': 50}    # Preferred value.

nn_params = optimize.minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
