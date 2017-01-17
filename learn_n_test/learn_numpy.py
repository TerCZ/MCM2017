import numpy as np
import math


# 基本操作
a = np.array([[1, 3, 5, 9],
              [4, 5, 6, 1]], dtype=np.complex)
print("element type is", a.dtype)
print("array shape is", a.shape)
print(a)

# 修改.shape属性只是改变访问方式，内存排列不会改变
a.shape = (4, 2)
print("change a.shape to (4, 2)")
print(a)

# arange()方法使用与标准函数range()类似
a = np.arange(0.3, 1.3, 0.24)
print("create array with np.arange")
print(a)

# 通过linspace()创建等差数列，endpoint表示是否包含结尾数值
a = np.linspace(0, 2*np.pi, num=4, endpoint=False)
print("create array with np.linspace")
print(a)

# 通过logspace()创建等比数列（log下的等差数列）
a = np.logspace(0, 3, num=4, endpoint=True)
print("create array with np.logspace(0, 3, 4)")
print(a)


# 通过自定义函数创建array
def func(i):
    return 5 * i ** 2

a = np.fromfunction(func, (2,))
print("create array with np.fromfunction")
print(a)

# 通过切片访问array
a = np.arange(10)
print("visit a with slice")
print(a[5::-1])

# 通过数列访问array
a[(1, 5, 6), ] = -1, -5, -6
print("visit a lije a[[1, 5, 6]]")
print(a)

# 数列访问得到新的对象与原始对象内存上是分离的，修改不会相互影响
b = a[[3, 4, -2]]
print("b = a[3, 4, -2[]]")
print(b)
b[1] = 100
print("set b[1] = 100, a is not effected")
print(a)
print(b)

# 通过布尔array（不是内置list）访问，返回array为对应位置值为True的元素
# 若使用布尔list，则会被解释为0、1编号
print("visit a with boolean array (not list)")
print(a[np.array([True, True, False, False, True, True, True, False, False, True])])
print("but if you use boolean list")
print(a[[True, True, False, False, True, True, True, False, False, True]])

# 生成布尔array的一种方案
a = np.random.rand(5)
print(a)
print(a > 0.5)
print(a[a > 0.5])

# 通过布尔array访问多位数组
a = np.arange(0, 13, 3).reshape(-1, 1) + np.arange(0, 3)
print(a)
print(a[(2, 4, 3), (0, 1, 2)])
print(a[:3, [0, 2]])
mask = np.random.rand(5) > 0.5
print(a[mask, :])

# 自定义元素类型
mytype = np.dtype({"names": ["name", "id", "address", "phone", "email"],
                   "formats": ["S64", "i", "S256", "S16", "S128"]})
a = np.array([("Ter", 448, "D01-423", "2731", "t.chuzhe"),
              ("David", 000, "Matthew", "0000", "david.m")], dtype=mytype)
a[:]["name"] = "hell"

# 使用通用函数（universal function）
a = np.sin(np.linspace(0, 2*np.pi, 5))
print(a)

a = np.random.randint(0, 10, 5)
b = np.random.randint(0, 10, 5)
print(a)
print(b)
print(a ** b)
print(a // b)


# 自定义生成函数
def func(x):
    return math.sin(x) ** 2

a = np.random.randint(0, 10, 5)
b = np.array([func(x) for x in a])

# 自定义通用函数
myfunc = np.frompyfunc(func, 1, 1)
c = myfunc(a)
c.astype(np.float)  # 生成array元素类型为Object，需手动转化
print(b)
print(c)

# 方便地生成行向量与列向量
x, y = np.ogrid[0:4:9j, 2:6:3]
print(x)
print(y)
print(x + y)

x, y = np.ogrid[-2:2:20j, -2:2:20j]
z = x * np.exp(-x**2 - y**2)
print(z)

# 默认为元素一一对应地运算，若需要使用矩阵运算，需显示生成
a = np.matrix(np.arange(9).reshape((3, 3)))
print(a)
a *= a
print(a)
a **= -1  # 求逆
print(a)

# 求解多元一次方程
a = np.random.rand(10, 10)
b = np.random.rand(10)
x = np.linalg.solve(a, b)
print(np.dot(a, x))  # 点乘，此外还提供inner(), outer()等运算
print(b)

# 储存为numpy专用二进制文件，便于python下存取，但不能跨语言使用
a = np.random.rand(5, 3)
np.save("a.npy", a)
b = np.load("a.npy")
print(a)
print(b)
