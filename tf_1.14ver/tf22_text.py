import tensorflow as tf
import numpy as np

# 1. 데이터 

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
total_batch = 6
learning_rate = 2e-2
                                                            #6            #4
X = tf.compat.v1.placeholder(tf.float32, shape = (None,sequence_length,input_dim))
x = tf.compat.v1.placeholder(tf.float32, shape = (None,1,4))

Y = tf.compat.v1.placeholder(tf.float32, shape = (1,6)) 
y = tf.compat.v1.placeholder(tf.float32, shape = (None,1)) 



# 2. 모델 구성 - hidden layer 없는 lstm 모델 

cell = tf.nn.rnn_cell.BasicLSTMCell(output)
hypothesis, _stats = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)

# 3. 컴파일
weights = tf.cast(tf.random_normal([1,6,4]), dtype = tf.float32) 

cost = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.compat.v1.train.AdamOptimizer(learning_rate= learning_rate).minimize(cost)

# 4. 훈련 

with tf.Session() as sess :  
    sess.run(tf.global_variables_initializer()) 
    
    for i in range(500) :
        
        # result = sess.run(prediction, feed_dict = {X:x_data})
        
        for i in range(total_batch) :   # 6
         
            start  = i * batch_size
            end = start + batch_size

            batch_xs, batch_ys = x_data[:, start : end], y_data[:,start : end]
            
            feed_dict = {x:batch_xs, y: batch_ys} 
            
            loss, _ = sess.run([cost, train], feed_dict = feed_dict )
              
             
            print(i , 'loss : ' , loss)    
    
        # result_str = [idx2char[c] for c in np.squeeze(result)]
        # print('\nPrediction str : ', ''.join(result_str))

