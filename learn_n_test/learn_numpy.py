import numpy as np
import math

# basic operation
# a = np.array([[1, 3, 5, 9],
#               [4, 5, 6, 1]], dtype=np.complex)
# print("element type is", a.dtype)
# print("array shape is", a.shape)
# print(a)

# a.shape = (4, 2)
# print("change a.shape to (4, 2)")
# print(a)

# a = np.arange(0.3, 1.3, 0.24)
# print("create array with np.arange")
# print(a)

# a = np.linspace(0, 2*np.pi, num=4, endpoint=False)
# print("create array with np.linspace")
# print(a)

# a = np.logspace(0, 3, num=4, endpoint=True)
# print("create array with np.logspace(0, 3, 4)")
# print(a)


# def func(i):
#     return 5 * i ** 2
# 
# a = np.fromfunction(func, (2,))
# print("create array with np.fromfunction")
# print(a)

# a = np.arange(10)
# print("visit a with slice")
# print(a[5::-1])
# a[(1, 5, 6),] = -1, -5, -6
# print("visit a lije a[[1, 5, 6]]")
# print(a)
# b = a[[3, 4, -2]]
# print("b = a[3, 4, -2[]]")
# print(b)
# b[1] = 100
# print("set b[1] = 100, a is not effected")
# print(a)
# print(b)
# print("visit a with boolean array (not list)")
# print(a[np.array([True, True, False, False, True, True, True, False, False, True])])
# print("but if you use boolean list")
# print(a[[True, True, False, False, True, True, True, False, False, True]])

# a = np.random.rand(5)
# print(a)
# print(a > 0.5)
# print(a[a > 0.5])

# a = np.arange(0, 13, 3).reshape(-1, 1) + np.arange(0, 3)
# print(a)
# print(a[(2, 4, 3), (0, 1, 2)])
# print(a[:3, [0, 2]])
# mask = np.random.rand(5) > 0.5
# print(a[mask, :])

# customized data type
# mytype = np.dtype({"names": ["name", "id", "address", "phone", "email"],
#                    "formats": ["S64", "i", "S256", "S16", "S128"]})
# a = np.array([("Ter", 448, "D01-423", "2731", "t.chuzhe"),
#               ("David", 000, "Matthew", "0000", "david.m")], dtype=mytype)
# a[:]["name"] = "hell"

# universal function
# a = np.linspace(0, 2*np.pi, 5)
# b = np.sin(a)

# a = np.random.randint(0, 10, 5)
# b = np.random.randint(0, 10, 5)
# print(a)
# print(b)
# print(a ** b)
# print(a // b)


# def func(x):
#     return math.sin(x) ** 2
#
# a = np.random.randint(0, 10, 5)
# b = np.array([func(x) for x in a])
# myfunc = np.frompyfunc(func, 1, 1)
# c = myfunc(a)
# c.astype(np.float)
# print(b)
# print(c)

# x, y = np.ogrid[0:4:9j, 2:6:3]
# print(x)
# print(y)
# print(x + y)

# x, y = np.ogrid[-2:2:20j, -2:2:20j]
# z = x * np.exp( - x**2 - y**2)
# print(z)

# a = np.matrix(np.arange(9).reshape((3, 3)))
# print(a)
# a *= a
# print(a)
# a **= -1
# print(a)

# a = np.random.rand(10, 10)
# b = np.random.rand(10)
# x = np.linalg.solve(a, b)
# print(np.dot(a, x))
# print(b)

# a = np.random.rand(5, 3)
# np.save("a.npy", a)
# b = np.load("a.npy")
# print(a)
# print(b)
