import numpy as np

ones = np.ones((10, 1))
y = np.matrix(ones)
print("ones shape: ", np.shape(ones))
print("y shape: ", np.shape(y))
for i in range(10):
    y = np.concatenate((y, ones), axis=1)

print(y)
