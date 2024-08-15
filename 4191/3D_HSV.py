import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



# 假设你的数据保存在 output.out 文件中
file_name = 'output.out'

# 读取文件中的数组
data = np.loadtxt(file_name)

# 提取 x, y, z 坐标
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# 包围框的上下界 (可以根据你的数据点进行调整)
boxes = [
    {'min': [25, 70, 0], 'max': [60, 120, 255]},
    {'min': [25, 20, 50], 'max': [60, 255, 160]},
    {'min': [0, 0, 220], 'max': [72, 100, 255]}
]

def draw_bounding_box(ax, min_bound, max_bound):
    # 定义长方体的8个顶点
    vertices = [
        [min_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], max_bound[1], max_bound[2]],
        [min_bound[0], max_bound[1], max_bound[2]]
    ]
    
    # 定义长方体的6个面（每个面由4个顶点组成）
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
        [vertices[4], vertices[7], vertices[3], vertices[0]]   # 左面
    ]
    
    # 绘制长方体的面
    poly3d = Poly3DCollection(faces, alpha=0.25, linewidths=1, edgecolors='r')
    poly3d.set_facecolor([0.5, 0.5, 1, 0.1])  # 设置面颜色和透明度
    ax.add_collection3d(poly3d)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点
ax.scatter(x, y, z)

# 绘制包围框
for box in boxes:
    draw_bounding_box(ax, box['min'], box['max'])

# 设置坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 设置坐标轴范围
ax.set_xlim([0, 255])
ax.set_ylim([0, 255])
ax.set_zlim([0, 255])

plt.show()

