from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img


# 데이터화
img_dog = load_img('data\dog_cat\dog.jpg', target_size= (224,224))
img_cat = load_img('data\dog_cat\cat.jpg', target_size= (224,224))
img_suit = load_img('data\dog_cat\suit.jpg', target_size= (224,224))
img_yang = load_img('data\dog_cat\yang.jpg', target_size= (224,224))


# plt.imshow(img_dog)
# plt.show()





# 하지만 이미지를 가져온 것 만으로는 연산을 하지 못한다 
# 그래서 numpy array화 시켜준다 
from keras.preprocessing.image import img_to_array

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_suit = img_to_array(img_suit)
arr_yang = img_to_array(img_yang)

# print(arr_dog)
# print(type(arr_dog))  # <class 'numpy.ndarray'>
# print(arr_dog.shape)  # (224, 224, 3)




# VGG16 쓰기위해 RGB -> BGR 변환
from keras.applications.vgg16 import preprocess_input

arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_suit = preprocess_input(arr_suit)
arr_yang = preprocess_input(arr_yang)

# print(arr_dog)  


# 이미지를 하나로 합치기 
import numpy as np

arr_input = np.stack([arr_dog, arr_cat, arr_suit, arr_yang])  
print(arr_input.shape)  # (4, 224, 224, 3)


# 모델 구성
model = VGG19()
probs = model.predict(arr_input)

print(probs)
# print('probs.shape :' , probs.shape)  # probs.shape : (4, 1000)

# 그런데 결과값이 이렇게 나온다
#[[3.1875824e-09 1.3608434e-09 5.5681160e-10 ... 2.8637179e-103.9494594e-08 9.8532462e-07]

# 이미지 결과
from keras.applications.vgg16 import decode_predictions

result = decode_predictions(probs)

print('=================')
print(result[0])
print('=================')
print(result[1])
print('=================')
print(result[2])
print('=================')
print(result[3])


'''
image net

print(result[0])
[('n02099601', 'golden_retriever', 0.89813465),   89% 확률로 골든 리트리버 
 ('n02099712', 'Labrador_retriever', 0.08291211), 
 ('n02104029', 'kuvasz', 0.0059011644), 
 ('n02111500', 'Great_Pyrenees', 0.004597914), 
 ('n02111129', 'Leonberg', 0.0012932649)]

print(result[1])
[('n02123045', 'tabby', 0.4093532), 
 ('n02123159', 'tiger_cat', 0.35503846), 
 ('n02124075', 'Egyptian_cat', 0.15406536), 
 ('n02127052', 'lynx', 0.010960355), 
 ('n04522168', 'vase', 0.0047452184)]

print(result[2])
[('n04591157', 'Windsor_tie', 0.32979473), 
 ('n04350905', 'suit', 0.27874398), 
 ('n02883205', 'bow_tie', 0.1355967), 
 ('n04599235', 'wool', 0.06498963), 
 ('n04371430', 'swimming_trunks', 0.018251115)]      

print(result[3])
('n03000247', 'chain_mail', 0.19996196),  체인메일...? 갑옷?
('n03866082', 'overskirt', 0.18410926), 
('n03980874', 'poncho', 0.12426462), 
('n03534580', 'hoopskirt', 0.064331114), 
('n03877472', 'pajama', 0.05618867)] 

'''