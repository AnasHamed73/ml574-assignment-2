import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    activition = 1/(1+np.exp(-1*z))
    return activition # your code here


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    



    print('preprocess done')
    test0 = mat.get('test0')
    test1 = mat.get('test1')
    test2 = mat.get('test2')
    test3 = mat.get('test3')
    test4 = mat.get('test4')
    test5 = mat.get('test5')
    test6 = mat.get('test6')
    test7 = mat.get('test7')
    test8 = mat.get('test8')
    test9 = mat.get('test9')


    test0_label = np.zeros((len(test0)))
    test1_label = np.ones((len(test1)))
    test2_label = np.ones((len(test2)))
    test2_label.fill(2)
    test3_label = np.ones((len(test3)))
    test3_label.fill(3)
    test4_label = np.ones((len(test4)))
    test4_label.fill(4)
    test5_label = np.ones((len(test5)))
    test5_label.fill(5)
    test6_label = np.ones((len(test6)))

    test6_label.fill(6)
    test7_label = np.ones((len(test7)))
    test7_label.fill(7)
    test8_label = np.ones((len(test8)))
    test8_label.fill(8)
    test9_label = np.ones((len(test9)))
    test9_label.fill(9)

    test0 = np.column_stack((test0, test0_label))
    test1 = np.column_stack((test1, test1_label))
    test2 = np.column_stack((test2, test2_label))
    test3 = np.column_stack((test3, test3_label))
    test4 = np.column_stack((test4, test4_label))
    test5 = np.column_stack((test5, test5_label))
    test6 = np.column_stack((test6, test6_label))
    test7 = np.column_stack((test7, test7_label))
    test8 = np.column_stack((test8, test8_label))
    test9 = np.column_stack((test9, test9_label))

    test = np.array(np.vstack((test0,test1,test2,test3,test4,test5,test6,test7,test8,test9)))

    test_new = test
    test_data = test_new[:,0:784]
    test_label = test_new[:,784]
    np.true_divide(test,255.0)
    #print(mat)

    train0 = mat.get('train0')
    train1 = mat.get('train1')
    train2 = mat.get('train2')
    train3 = mat.get('train3')
    train4 = mat.get('train4')
    train5 = mat.get('train5')
    train6 = mat.get('train6')
    train7 = mat.get('train7')
    train8 = mat.get('train8')
    train9 = mat.get('train9')
    train0_label = np.zeros((len(train0)))
    train1_label = np.ones((len(train1)))
    train2_label = np.ones((len(train2)))
    train2_label.fill(2)
    train3_label = np.ones((len(train3)))
    train3_label.fill(3)
    train4_label = np.ones((len(train4)))
    train4_label.fill(4)
    train5_label = np.ones((len(train5)))
    train5_label.fill(5)
    train6_label = np.ones((len(train6)))
    train6_label.fill(6)
    train7_label = np.ones((len(train7)))
    train7_label.fill(7)
    train9_label = np.ones((len(train9)))
    train9_label.fill(9)
    train8_label = np.ones((len(train8)))
    train8_label.fill(8)

    train0 = np.column_stack((train0, train0_label))
    train1 = np.column_stack((train1, train1_label))
    train2 = np.column_stack((train2, train2_label))
    train3 = np.column_stack((train3, train3_label))
    train4 = np.column_stack((train4, train4_label))
    train5 = np.column_stack((train5, train5_label))
    train6 = np.column_stack((train6, train6_label))
    train7 = np.column_stack((train7, train7_label))
    train8 = np.column_stack((train8, train8_label))
    train9 = np.column_stack((train9, train9_label))
    train = np.array(np.vstack((train0, train1, train2, train3, train4, train5, train6, train7, train8, train9)))


    np.random.shuffle(train)



    train_new = train[0:50000,:]
    train_label = train_new[:, 784]
    train_data = train_new[:, 0:784]


    validation_new = train[50000:60000,:]
    validation_data = validation_new[:,0:784]
    validation_label = validation_new[:,784]

    np.true_divide(test_data, 255.0)
    np.true_divide(validation_data,255.0)


    # Feature selection
    # Your code here.
    all_number = np.vstack((train_data, test_data,validation_data))


    first_row = np.sum(all_number[:,0])
    while first_row == 0:
        all_number = np.delete(all_number,0,1)
        first_row = np.sum(all_number[:, 0])

    last_row = np.sum(all_number[:,-1])
    while last_row == 0:
        all_number = np.delete(all_number,-1,1)
        last_row == np.sum(all_number[:,-1])



    train_data = all_number[0:len(train_data),:]
    test_data = all_number[len(train_data):len(train_data)+len(test_data),:]
    validation_data = all_number[len(train_data)+len(test_data):len(train_data)+len(test_data)+len(validation_data),:]












    return train_data, train_label, validation_data, validation_label, test_data, test_label


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

    # Your code here
    #
    #
    #
    #
    #



    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image

    % Output: 
    % label: a column vector of predicted labels"""

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

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
