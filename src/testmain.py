import numpy as np

npa = np.array([[1, 2, 3], [4, 5, 6]])
os = npa.shape
print(npa)
print(os)
a = tuple(npa.reshape(-1))
print(a)

a = np.array(a).reshape(os)
print(a)
print(a.shape)
