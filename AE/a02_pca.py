import numpy as np
from sklearn.decomposition import PCA 
from sklearn.datasets import load_diabetes 

dataset = load_diabetes()

X = dataset.data
Y = dataset.target 

print(X.shape) # (442, 10)
print(Y.shape) # (442,)

pca = PCA(n_components=5)

x2 = pca.fit_transform(X)
pca_evr = pca.explained_variance_ratio_
print(pca_evr)
# [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856]

print(sum(pca_evr))
# 0.8340156689459766 -> 17%의 손실 

# 압축한 컬럼들에 대한 중요도 
# 각각의 주성분 벡터가 이루는 축에 투영(projection)한 결과의 분산의 비율
# 첫번째 차원의 중요도 40%, 두번쨰 차원은 14% ..이렇게 나가므로 
# n_component =2 로 주면 54%의 보존율을 보인다..보일까...?

# PCA는 원래 데이터셋(original dataset)에서 특정 Feature를 선택하는 것이 아니라, 
# Feature들이 구성하는 차원(dimension)에서의 부분 공간(subspace)를 구성하는 축(eigenvector)를 찾아 
# 투영(projection)해주는 알고리즘

# 힌 번 원형 탈수기를 돌리고 (강도 = n_comp) 그 뒤에 남은 것들의 비율
# 무겁고 중심에 가까운 것들이 남는다 

# 효율을 위한 것이지 성능,정확도를 높인다는 의도는 아님 

#  feature_importance -> 전체 컬럼들에 대한 중요도

