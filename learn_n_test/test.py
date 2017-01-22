import numpy as np

a = np.array([1] * 8)
a *= 2
a += np.array([3] * 4)
print(a)