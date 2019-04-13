'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

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
    return 1/(1+np.exp(-1*z))  # your code here
    
    
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    label_matrix = []
    for label in training_label:
        label_matrix.append([0 if x != label else 1 for x in range(2)])
    label_matrix = np.array(label_matrix)

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Feedforward Propagation
    bias_1 = np.ones((training_data.shape[0], 1))
    training_data_with_bias = np.concatenate((training_data, bias_1), axis=1)
    hidden_output = sigmoid(np.dot(training_data_with_bias, w1.T))

    bias_2 = np.ones((1, hidden_output.T.shape[1]))
    hidden_output_with_bias = np.concatenate((hidden_output.T, bias_2), axis=0).T
    Feedforward_output = sigmoid(np.dot(hidden_output_with_bias, w2.T))

    # obj_val

    # ff = (np.log(Feedforward_output)).T
    # gg = (np.log(1 - Feedforward_output)).T
    # J_sum = np.array(-(np.dot(label_matrix, ff) + np.dot((1 - label_matrix), gg)))

    # diagonal = np.zeros((np.shape(J_sum)[0], 1))
    # for i in range(np.shape(J_sum)[0]):
    #    diagonal[i][0] = J_sum[i][i]
    # J_w1_w2 = np.mean(diagonal, axis=0)
    A = np.ones((np.shape(label_matrix)[0], 2))
    Item_1 = np.multiply(label_matrix, np.log(Feedforward_output))
    Item_2 = np.multiply(A - label_matrix, np.log(A - Feedforward_output))
    M = Item_1 + Item_2
    J_w1_w2 = (-1 * np.sum(np.sum(M, axis=0)))/training_data.shape[0]

    w1_squared = np.dot(w1.flatten(), w1.flatten().T)
    w2_squared = np.dot(w2.flatten(), w2.flatten().T)
    regularization_term = np.dot(lambdaval, (w1_squared + w2_squared))
    obj_val = J_w1_w2.flatten() + regularization_term / np.dot(2, training_data.shape[0])

    # Error function and Backpropagation
    delta_l = np.array(label_matrix - Feedforward_output) # correspondes to eqn(9)
    dev_lj = -1*np.dot(delta_l.T, hidden_output_with_bias) # correspondes to eqn(8)
    grad_w2 = (dev_lj + lambdaval *w2)/ training_data.shape[0] #correspondes to eqn(16)
    w2_noBias = w2[:,0:-1]
    delta_j = np.array(hidden_output)*np.array(1-hidden_output) # correspondes to -(1-Zj)Zj in eqn(12)
    dev_ji = -1*np.dot((np.array(delta_j)*np.array(np.dot(delta_l,w2_noBias))).T,training_data_with_bias) # correspondes to eqn(12)
    grad_w1 = (dev_ji+lambdaval*w1)/training_data.shape[0] #correnspondes to eqn(17)

    # Reshape the gradient matrices to a 1D array.
    grad_w1_reshape = np.ndarray.flatten(grad_w1.reshape((grad_w1.shape[0]*grad_w1.shape[1],1)))
    grad_w2_reshape = grad_w2.flatten()
    obj_grad_temp = np.concatenate((grad_w1_reshape.flatten(), grad_w2_reshape.flatten()),0)
    obj_grad = np.ndarray.flatten(obj_grad_temp)
    return (obj_val,obj_grad)


# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    input_bias = np.ones((data.shape[0],1))  # create a bias
    data_bias = np.concatenate((data, input_bias), axis=1)  # add bias to training data
    hiden_out = sigmoid(np.dot(data_bias, w1.T))  # 3.32 equtions 1 and 2
    hiden_bias = np.ones((1,hiden_out.T.shape[1]))  # create a bias
    hiden_out_bias = np.concatenate((hiden_out.T, hiden_bias), axis=0).T  # add bias to hidden_out data
    net_out = sigmoid(np.dot(hiden_out_bias,w2.T))  # 3.32 eqution 3 and 4, feed forward is complete.
    # Make a 1D vector of the predictions.
    return net_out.argmax(axis=1)



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
n_hidden = 20 
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 25;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
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
