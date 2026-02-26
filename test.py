from scipy.io import loadmat

mat = loadmat("data/traj/run1_car_route_gt_1.mat")
print(mat.keys())          # 看里面有哪些变量名
# x = mat["x"]               # 取出变量 x（一般是 numpy 数组）