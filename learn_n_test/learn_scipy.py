import math
import numpy as np
import scipy.optimize as opt
from scipy import interpolate
from scipy import integrate
import sympy
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# 非线性函数最小二乘法拟合
# 定义所求函数
def func(x, p):
    A, k, theta = p
    return A * np.sin(2*np.pi*k*x + theta)


# 定义偏差计算函数
def residuals(p, y, x):
    return y - func(x, p)

x = np.linspace(0, -2*np.pi, 100)  # 注意，x范围为[0, -2*pi]
A, k, theta = 10, 0.34, np.pi/6
y0 = func(x, (A, k, theta))
y1 = y0 + 2 * np.random.randn(len(x))  # 手动添加噪声

p0 = [7, 0.2, 0]  # 初始猜测的值
plsq = opt.leastsq(residuals, p0, args=(y1, x))
print([A, k, theta])
print(plsq[0])

pl.plot(x, y0, label="真实曲线")
pl.plot(x, y1, label="噪声曲线")
pl.plot(x, func(x, plsq[0]), label="拟合曲线")
pl.legend()
pl.show()


# 求解非线性方程
# 定义误差函数
def f(x):
    x0 = float(x[0])
    x1 = float(x[1])
    x2 = float(x[2])
    return [
        5 * x1 + 3,
        4 * x0 ** 2 - 2 * math.sin(x1 * x2),
        x1 * x2 - 1.5]

result = opt.fsolve(f, [1, 2, 3])  # 1，2，3为猜测初始值
print(result)
print(f(result))

# B-Spline样条曲线
x = np.linspace(0, 2*np.pi + np.pi/4, 10)
y = np.sin(x)
x_new = np.linspace(0, 2*np.pi + np.pi/4, 100)
f_linear = interpolate.interp1d(x, y)  # 得到线性插值
tck = interpolate.splrep(x, y)  # 算出B-Spline曲线的参数
y_bspline = interpolate.splev(x_new, tck)  # 计算插值结果

pl.plot(x, y, "o",  label="原始数据")
pl.plot(x_new, f_linear(x_new), label="线性插值")
pl.plot(x_new, y_bspline, label="B-spline插值")
pl.legend()
pl.show()


# 一元数值积分
def half_circle(x):
    return (1 - x**2) ** 0.5

N = 10000
x = np.linspace(-1, 1, N, endpoint=False)
dx = 2 / N
y = half_circle(x)
print(2 * dx * np.sum(y))
print(2 * np.trapz(y, x))
pi_half, err = integrate.quad(half_circle, -1, 1)   # 使用quad()积分精度提高很多
print(pi_half * 2, err)


# 二元数值积分
def half_sphere(x, y):
    return np.sqrt(1 - x**2 - y**2)

# dblquad(func2d, a, b, gfun, hfun)
result, err = integrate.dblquad(half_sphere, -1, 1, lambda x: -half_circle(x), half_circle)
print(3 / 2 * result, err)


# 解常微分方程
# 返回值分别对应dx/dt, dy/dt, dz/dt
def lorenz(w, t, p, r, b):
    x, y, z = w
    return np.array([
        p * (y - x),
        x * (r - z) - y,
        x * y - b * z])

t = np.arange(0, 30, 0.01)
track1 = integrate.odeint(lorenz, (0, 1, 0), t, args=(10, 28, 3))
track2 = integrate.odeint(lorenz, (0, 1.01, 0), t, args=(10, 28, 3))

fig = plt.figure()
ax = Axes3D(fig)
ax.plot(track1[:, 0], track1[:, 1], track1[:, 2])
ax.plot(track2[:, 0], track2[:, 1], track2[:, 2])
plt.show()

# 符号计算
# 欧拉公式
x = sympy.Symbol("x", real=True)
print(sympy.expand(sympy.E**(sympy.I*x), complex=True))

# 级数展开
sympy.pprint(sympy.series(sympy.exp(sympy.I*x), x, 0, 10))

# 不定积分
print(sympy.integrate(x*sympy.sin(x), x))

# 定积分
print(sympy.integrate(x*sympy.sin(x), (x, 0, 2*sympy.pi)))

# 定积分解球体体积
x, y = sympy.symbols("x, y")
r = sympy.Symbol("r", positive=True)
circle_erea = 2 * sympy.integrate(sympy.sqrt(r**2 - x**2), (x, -r, r))
print(circle_erea)
circle_erea = circle_erea.subs(r, sympy.sqrt(r**2 - x**2))
print(circle_erea)
sphere_volume = sympy.integrate(circle_erea, (x, -r, r))
print(sphere_volume)

# 无穷的表示
print(sympy.integrate(1 / x**2, (x, 1, sympy.oo)))
