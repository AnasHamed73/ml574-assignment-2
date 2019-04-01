import numpy as np
def sigmoid(z):
    return  1/(1+np.exp(z))
b=np.array([[2,3,2,3],[2,3,5,5]])
c=b.shape[0]
d=np.array([3,4])
f=np.array([5,6])
a=sigmoid(b)
g = np.ones((b.shape[0],1))
#f=np.concatenate((b,d))
#print(np.dot((np.ones(b.shape)-b).flatten(),(np.log(np.ones(b.shape)-b).flatten().T))
print(b[:,1])
