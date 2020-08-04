import numpy as np
from sklearn.decomposition import PCA 
from sklearn.datasets import load_diabetes 

dataset = load_diabetes()

X = dataset.data
Y = dataset.target 

print(X.shape) # (442, 10)
print(Y.shape) # (442,)

# pca = PCA(n_components=5)

# x2 = pca.fit_transform(X)
# pca_evr = pca.explained_variance_ratio_
# print(pca_evr)
# print(sum(pca_evr))

pca = PCA()
pca.fit(X)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)
# cumsum 해당 축의 누적합
# [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759 0.94794364 0.99131196 0.99914395 1.        ]

a = np.argmax(cumsum >= 0.94) + 1

# print(cumsum >= 0.94)
# 조건문 [False False False False False False  True  True  True  True]

print(a) # 7