import sympy as sp
import numpy as np

#两个三维向量，分别计算两个向量的模
a = np.array([10,2,3])
b = np.array([-7,-5,-2])
a_mod = np.linalg.norm(a)
b_mod = np.linalg.norm(b)
print("a模：", a_mod)
print("b模：", b_mod)
#计算两个向量的点乘
dot_product = np.dot(a, b)
print("a与b的点乘：", dot_product)
#计算两个向量的夹角
cos_angle = dot_product / (a_mod * b_mod)
angle = np.arccos(cos_angle)
print("a与b的夹角：", angle)
#计算两个向量的叉乘
cross_product = np.cross(a, b)
print("a与b的叉乘：", cross_product)
print("a与b的叉乘的模：", np.linalg.norm(cross_product))