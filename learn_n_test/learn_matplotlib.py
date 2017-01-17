import numpy as np
import matplotlib.pyplot as plt


# 一个简单图例
x = np.linspace(0, 10, 1000)
y1 = np.sin(x)
y2 = np.cos(x**2)

plt.figure(figsize=(8, 4))  # 单位为英尺，dpi默认为80，故该图像尺寸为640*320像素
plt.plot(x, y1, label="$sin(x)$", color="red", linewidth=2)
plt.plot(x, y2, "b--", label="$cos(x^2)$")  # label使用"$$"令matplotlib调用latex引擎
plt.xlabel("Time (s)")
plt.ylabel("Volt")
plt.title("Example")
plt.ylim(-1.2, 1.2)
plt.legend()
plt.show()

# 配置属性
x = np.arange(0, 10, 0.1)
line, = plt.plot(x, x*x)
line.set_antialiased(False)  # 关闭反锯齿
lines = plt.plot(x, np.sin(x), x, np.cos(x))  # 同时画两条曲线
plt.setp(lines, color="r", linewidth=2)

print(line.get_linewidth())
print(plt.getp(lines[0], "color"))
print(plt.getp(lines[1]))
print(plt.getp(plt.gcf()))
print(plt.getp(plt.gcf(), "axes"))
print(plt.gca())

# 多轴图
for idx, color in enumerate("rgbyck"):
    plt.subplot(320+idx+1, axisbg=color)
plt.show()

# 横跨多列多行
plt.subplot(241)
plt.subplot(242)
plt.subplot(223)
plt.subplot(143)
plt.subplot(144)
plt.show()

# Artist对象
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # 图像在窗口左、下相对位置与横、高的相对长度
line, = ax.plot([1, 2, 3], [2, 1, 4])
plt.show()

# 一些Artist属性
# alpha : 透明度，值在0到1之间，0为完全透明，1为完全不透明
# animated : 布尔值，在绘制动画效果时使用
# axes : 此Artist对象所在的Axes对象，可能为None
# clip_box : 对象的裁剪框
# clip_on : 是否裁剪
# clip_path : 裁剪的路径
# contains : 判断指定点是否在对象上的函数
# figure : 所在的Figure对象，可能为None
# label : 文本标签
# picker : 控制Artist对象选取
# transform : 控制偏移旋转
# visible : 是否可见
# zorder : 控制绘图顺序
