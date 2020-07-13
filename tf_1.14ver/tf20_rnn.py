import tensorflow as tf
import numpy as np



# 1. 데이터 =  hihello 

idx2char = [ 'e', 'h', 'i', 'l' ,'o']


_data = np.array([['h', 'i', 'h', 'e', 'l','l','o']], dtype= np.str).reshape(-1,1)
# print(_data.shape)   (7, 1)
# print(_data) 
# print(type(_data))  

# One - Hot Encoding 
# e = [ 0 1 0 0 0 0]
# h = [ 0 0 1 0 0 0]
# . . . 
from sklearn.preprocessing import OneHotEncoder 

ohe = OneHotEncoder()
# ohe.fit(_data)
# _data = ohe.transform(_data).toarray()
_data = ohe.fit_transform(_data).toarray()

# print("===============")
# print(_data)
# [[0. 1. 0. 0. 0.] h
#  [0. 0. 1. 0. 0.] i                                                                           
#  [0. 1. 0. 0. 0.] h      
#  [1. 0. 0. 0. 0.] e
#  [0. 0. 0. 1. 0.] l
#  [0. 0. 0. 1. 0.] l
#  [0. 0. 0. 0. 1.]]0

# print(type(_data))  <class 'numpy.ndarray'>
# print(_data.dtype)  float64


# -> 여기서 x = hihell, y = ihello 로 둔다 
x_data = _data[:6, : ]
y_data = _data[1:, : ]

y_data = np.argmax(y_data, axis = 1)
# print('==============')
# print(y_data)  [2 1 0 3 3 4]
# print(y_data.shape)  (6,)  

x_data = x_data.reshape(1,6,5)
y_data = y_data.reshape(1,6)


sequence_length = 6
input_dim = 5
output = 5
battch_size = 1 # 전체행
learning_rate = 0.1

X = tf.compat.v1.placeholder(tf.float32, shape = (None,sequence_length,input_dim))
# Y = tf.compat.v1.placeholder(tf.float32, shape = (None,input_dim))
Y = tf.compat.v1.placeholder(tf.int32, shape = (None,6)) # argmax로 한게 



# 2. 모델 구성 - hidden layer 없는 lstm 모델 
# model.add(LSTM(output, input_shape = (6,5))
cell = tf.nn.rnn_cell.BasicLSTMCell(output)
hypothesis, _stats = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
# print(hypothesis)  shape=(?, 6, 5개), dtype=float32), outout cell = 5개

# 3. 컴파일
weights = tf.ones([battch_size, sequence_length]) # 선형을 default로 잡음

sequence_loss = tf.contrib.seq2seq.sequence_loss(
        logits = hypothesis , targets = Y, weights = weights)

cost = tf.reduce_mean(sequence_loss)

train = tf.compat.v1.train.AdamOptimizer(learning_rate= learning_rate).minimize(cost)
prediction = tf.argmax(hypothesis, axis = 2)


# 4. 훈련 

with tf.Session() as sess :  
    sess.run(tf.global_variables_initializer()) 
    
    for i in range(401) :
        loss = sess.run([cost, train], feed_dict = {X : x_data, Y : y_data})
        result = sess.run(prediction, feed_dict = {X:x_data})
        
        print(i , 'loss : ' , loss, 'prediction :' ,result, "True Y : ", y_data)
        
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print('\nPrediction str : ', ''.join(result_str))
        
        