import tensorflow as tf
import numpy as np



# 1. 데이터 =  hihello 


dataset = np.array([1,2,3,4,5,6,7,8,9,10])

def split_x (seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1 ) :
        subset = seq[ i: (i + size)]
        aaa.append([item for item in subset])   
    print(type(aaa))
    return np.array(aaa)


# print(split_x(dataset,5))
# [[ 1  2  3  4  5]        
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]

data = split_x(dataset,5)

x_data = data[ :, :-1]
y_data = data[ : , -1]

x_data = x_data.reshape(1,6,4)
y_data = y_data.reshape(1,6)


sequence_length = 6 
input_dim = 4
output = 6
batch_size = 1 
learning_rate = 0.1

X = tf.compat.v1.placeholder(tf.float32, shape = (None,sequence_length,input_dim))
# Y = tf.compat.v1.placeholder(tf.float32, shape = (None,input_dim))
Y = tf.compat.v1.placeholder(tf.float32, shape = (1,6)) 



# 2. 모델 구성 - hidden layer 없는 lstm 모델 

cell = tf.nn.rnn_cell.BasicLSTMCell(output)
hypothesis, _stats = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)


# 3. 컴파일
weights = tf.random_normal([batch_size, sequence_length]) 

# sequence_loss = tf.contrib.seq2seq.sequence_loss(
#         logits = hypothesis , targets = Y, weights = weights)

cost = -tf.reduce_mean( Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

# cost = tf.reduce_mean(sequence_loss)

train = tf.compat.v1.train.AdamOptimizer(learning_rate= learning_rate).minimize(cost)
# prediction = tf.argmax(hypothesis, axis = 2)


# 4. 훈련 

with tf.Session() as sess :  
    sess.run(tf.global_variables_initializer()) 
    
    for i in range(401) :
        loss = sess.run([cost, train], feed_dict = {X : x_data, Y : y_data})
        # result = sess.run(prediction, feed_dict = {X:x_data})
        
        print(i , 'loss : ' , loss, 'prediction :' , hypothesis, "True Y : ", y_data)
        
        # result_str = [idx2char[c] for c in np.squeeze(result)]
        # print('\nPrediction str : ', ''.join(result_str))

