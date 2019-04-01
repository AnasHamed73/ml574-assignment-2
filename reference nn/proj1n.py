from math import sqrt
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
import pickle



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

    return  1/(1+np.exp(-z))


def preprocess():
    """
    Cleans handwritten digit data in mnist_all.mat and returns feature sets and labels.

    Returns:
        numpy.array: "train_data", shape (50000, 748), feature sets for all training data examples.
        numpy.array: "train_label", shape (50000,) true labels for training data examples.
        numpy.array: "validation_data", shape (10000, 784), feature sets for all validation data examples.
        numpy.array: "validation_label", shape (10000,) true labels for validation data examples.
        numpy.array: "test_data", shape (10000, 784), feature sets for all test data examples.
        numpy.array: "test_label", shape (10000,) true labels for test data examples.
    """
    # # Define some constants to make the code more readable and assist debugging.
    # FEATURE_LENGTH = 784  # This "constant" will actually change after simple feature selection.
    # EXPECTED_TRAINING = 60000
    # EXPECTED_VALIDATION = 10000
    # EXPECTED_TESTING = 10000
    # DEV_TRAINING = 500
    # DEV_VALIDATING = 100
    # DEV_TESTING = 100
    # DEBUG = False
    # if DEBUG:
    #     print ("OPERATING IN DEBUG MODE WITH REDUCED DATA SET!")
    #
    # # Read the matlab matrix file and store it as a 2D array in nested Python lists.
    # mat = loadmat('mnist_all.mat')
    # if DEBUG:
    #     print ("There are a total of", len(mat), "items in the .mat file:", mat.keys())
    # training_sets = []  # Should be a 2D list, 60000 x 784 elements long.
    # testing_sets = []  # Should be a 2D list, 60000 x 784 elements long.
    # all_60000_training_labels = []  # Should be a 1D list, 60000 (EXPECTED TRAINING) elements long.
    # test_label = []  # Should be a 1D list, 10000 (EXPECTED TESTING) elements long.
    # for k in mat:
    #     if k.find("test") != -1:  # It's one of test0 - test9.
    #         true_label = int(k[-1])
    #         for m in mat[k]:
    #             test_label.append(true_label)
    #             testing_sets.append(m)
    #     elif k.find("train") != -1:  # It's one of train0 - train9.
    #         true_label = int(k[-1])
    #         for m in mat[k]:
    #             all_60000_training_labels.append(true_label)
    #             training_sets.append(m)
    #     else:
    #         pass # Otherwise it's meta-information about the Matlab file that we don't need.
    #
    # # Put the 2D testing and training lists into numpy arrays and normalize them.
    # test_data = np.array(testing_sets, dtype='double') / 255
    # all_training = np.array(training_sets, dtype='double') / 255
    # test_label = np.array(test_label)
    # all_60000_training_labels = np.array(all_60000_training_labels)
    #
    # # Check that the data is in the form we expect.
    # assert test_label.shape == (EXPECTED_TESTING,), test_label.shape
    # assert all_training.shape == (EXPECTED_TRAINING, FEATURE_LENGTH)
    # assert test_data.shape == (EXPECTED_TESTING, FEATURE_LENGTH)
    # assert isinstance(all_training[0][0], np.float64)
    #
    # # Implement very simple feature selection,eliminate any features that are identical for every example in training.
    # temp = all_training.T
    # varied_feature_indices = []
    # for row in range(len(temp)):
    #     if np.unique(temp[row:row+1, ::]).size > 1:
    #         varied_feature_indices.append(row)
    # # Update the constant of our feature length.
    # FEATURE_LENGTH = len(temp[varied_feature_indices])
    # # Update the training data to exclude useless features.
    # temp = temp[varied_feature_indices]
    # all_training = temp.T
    # # Update the testing data too to exclude useless features.
    # # Of course in the real world we could choose which features to eliminate from our test data,
    # # but then again in the real world, we would not have our test data!
    # temp = test_data.T
    # temp = temp[varied_feature_indices]
    # test_data = temp.T
    #
    #
    # # Now randomly divide the training data into two matrices: one with 50,000 rows, and another with 10,000.
    # # The 50,000-row matrix is "training" data, and the 10,000-row matrix is "validating" data.
    # # Split their labels at the same time, to make sure that the true labels are at the same index as their data.
    # random_training_indices = np.array(random.sample(list(range(EXPECTED_TRAINING)), 50000))
    # random_validating_indices = np.array(list(set(list(range(EXPECTED_TRAINING))) - set(random_training_indices)))
    # # See http://docs.scipy.org/doc/numpy-1.10.1/user/basics.indexing.html, section "Index Arrays", for an explanation of this syntax on a np array.
    # train_data = all_training[random_training_indices]
    # validation_data = all_training[random_validating_indices]
    # train_label = all_60000_training_labels[random_training_indices]
    # validation_label = all_60000_training_labels[random_validating_indices]
    # # Double check that data is in expected shape.
    # assert train_data.shape == (EXPECTED_TRAINING - EXPECTED_VALIDATION, FEATURE_LENGTH)
    # assert validation_data.shape == (EXPECTED_VALIDATION, FEATURE_LENGTH)
    # assert train_label.shape == (EXPECTED_TRAINING - EXPECTED_VALIDATION, )
    # assert validation_label.shape == (EXPECTED_VALIDATION, )
    #
    # if DEBUG:
    #     # Return a smaller subset of the data for faster development.
    #     # Note that these are shallow copies http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.copy.html.
    #     train_data = train_data[:DEV_TRAINING]
    #     train_label = train_label[:DEV_TRAINING]
    #     validation_data = validation_data[:DEV_VALIDATING]
    #     validation_label = validation_label[:DEV_VALIDATING]
    #     random_testing_indices = np.array(random.sample(list(range(EXPECTED_TESTING)), DEV_TESTING))
    #     test_data = test_data[random_testing_indices]
    #     test_label = test_label[random_testing_indices]
    # else:
    #     pass # Return the really large data matrices.
    # print ("Returning training data with shape (rows, cols):" + str(train_data.shape))
    # print ("Returning labels for training data with shape (rows, cols):" + str(train_label.shape))
    # print ("Returning validation data with shape (rows, cols):" + str(validation_data.shape))
    # print ("Returning labels for validation data with shape (rows, cols):" + str(validation_label.shape))
    # print ("Returning test data with shape (rows, cols):" + str(test_data.shape))
    # print ("Returning labels for test data with shape (rows, cols):" + str(test_label.shape))
    # print ("Data type of all entries in matrices is:" + str(type(all_training[0][0])))



    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here
    delColumns = []
   # print(n_input)
    Fs=[]
    for i in range(train_preprocess.shape[1]):
        if (train_data[:, i].max() - train_data[:, i].min()) == 0 and \
                        (validation_data[:, i].max() - validation_data[:, i].min()) == 0 and \
                        (test_data[:, i].max() - test_data[:, i].min()) == 0:
            delColumns.append(i)
	else:
            Fs.append(i)
            
    
    np.delete(train_data, delColumns, 1)
    np.delete(validation_data, delColumns, 1)
    np.delete(test_data, delColumns, 1)	
    print('preprocess done')
    return train_data, train_label, validation_data, validation_label, test_data, test_label, Fs

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
        label_matrix.append([0 if x != label else 1 for x in range(10)])
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
    first_term2 = np.dot((np.ones(label_matrix.shape)-label_matrix).flatten(),(np.log(np.ones(net_out.shape)-net_out).flatten().T))
    
    #first_term = np.dot((label_matrix).flatten(),(np.log(net_out)).flatten().T)+np.dot(np.array(1-label_matrix).flatten(),(np.log(np.array(1-net_out)).flatten().T)
    #s
    first_term = first_term1+first_term2
    second_term = lambdaval*(np.dot(w1.flatten(),w1.flatten().T)+np.dot(w2.flatten(),w2.flatten().T))
    obj_val = -1/(training_data.shape[0])*first_term + second_term / (2*training_data.shape[0]) # finish off eqn (15)



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
    input_bias = np.ones((data.shape[0],1))  # create a bias
    data_bias = np.concatenate((data, input_bias), axis=1)  # add bias to training data
    hiden_out = sigmoid(np.dot(data_bias, w1.T))  # 3.32 equtions 1 and 2
    hiden_bias = np.ones((1,hiden_out.T.shape[1]))  # create a bias
    hiden_out_bias = np.concatenate((hiden_out.T, hiden_bias), axis=0).T  # add bias to hidden_out data
    net_out = sigmoid(np.dot(hiden_out_bias,w2.T))  # 3.32 eqution 3 and 4, feed forward is complete.
    # Make a 1D vector of the predictions.

    return net_out.argmax(axis=1)


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label, selected_features = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output u5it
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
print(predicted_label)
# find the accuracy on Training Dataset
print(predicted_label)
print(train_label)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

obj = [selected_features, n_hidden, w1, w2, lambdaval]
# selected_features is a list of feature indices that you use after removing unwanted features in feature selection step

pickle.dump(obj, open('params.pickle', 'wb'))

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
