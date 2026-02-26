from scipy.io import loadmat

mat = loadmat("./data/traj/test/run1_car_route_gt_101.mat")
print(mat.keys())  # 看里面有哪些变量名
pts_coord = mat["pts_coord"]  # 取出变量 x（一般是 numpy 数组）
print(pts_coord)

mat1 = loadmat("./data/mall/floor5.mat")
print(mat1.keys())
v = mat1["v"]
print(v)
