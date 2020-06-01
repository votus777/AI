from sklearn.datasets import load_iris

import numpy as np

iris = load_iris()

# print(type(iris))    # <class 'sklearn.utils.Bunch'>

x_data = iris.data
y_data = iris.target

# print(type(x_data))   # <class 'numpy.ndarray'>
# print(type(y_data))   # <class 'numpy.ndarray'>

# numpy 형식은 저장이 가능하다

np.save('./data/iris_x.npy','arr=x_data')
np.save('./data/iris_y.npy','arr=y_data')    # data 폴더에 npy 파일 생성


x_data_load = np.load('./data/iris_x.npy')
y_data_load = np.load('./data/iris_y.npy')


print(type(x_data_load))   # <class 'numpy.ndarray'>
print(type(y_data_load))   # <class 'numpy.ndarray'>
print(x_data_load.shape)  # (150,4)
print(y_data_load.shape)   # (150, )
