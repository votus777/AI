import tensorflow as tf
import numpy as np

from keras.datasets import cifar10
from keras.utils import np_utils


# 데이터 

(x_train, y_train),(x_test,y_test) = cifar10.load_data()

print('x_train[0] : ', x_train[0])    # 32 X 32 공간 안에 0 ~ 255 까지의 숫자 배열
print('y_train[0] : ', y_train[0])

print(x_train.shape)  # (50000, 32, 32, 3)
print(y_train.shape)  # (50000, 1)
print(x_test.shape)   # (10000, 32, 32, 3)
print(y_test.shape)   # (10000, 1)



y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# 변수 설정 
learning_rate = 0.001 
training_epochs = 5000
batch_size = 100 
total_batch = int(len(x_train)/ batch_size) # 50000 / 100


x = tf.placeholder(tf.float32, shape=[None,32,32,3])
x_img = tf.reshape(x, [-1, 32, 32, 3])

y = tf.placeholder(tf.float32, shape=[None,10])
keep_prob = tf.placeholder(tf.float32)


# Layer 설정 
 
w1 = tf.get_variable("w1", shape=[3, 3, 3, 32])   #  == ConV2D( 32, (3,3), input_shape = (28,28,1)) //  1은 channel     
L1 = tf.nn.conv2d(x_img, w1, strides=[1,1,1,1], padding='SAME')   # stride 2로 하고 싶으면 [ 1, 2, 2, 1] 
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
L1 = tf.nn.dropout(L1, rate=  keep_prob)


print("w1 : ",w1)    # shape=(3, 3, 3, 32)
print("L1 : ",L1)    # shape=(?, 16, 16, 32)
 
w2 = tf.get_variable("w2", shape=[3, 3, 32, 64])    # 32 => channel 
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')   
L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
L2 = tf.nn.dropout(L2, rate=  keep_prob)


print("w2 : ",w2)    #  shape=(3, 3, 32, 64)
print("L2 : ",L2)    #  shape=(?, 8, 8, 64)

L2_flat = tf.reshape(L2, [-1, 8*8*64])

print('L2 flat : ', L2_flat)

w3 = tf.get_variable("W3", shape=[8*8*64, 512], 
                     initializer=tf.contrib.layers.xavier_initializer())   
b3 = tf.Variable(tf.zeros([512]))
L3 = tf.nn.selu(tf.matmul(L2_flat, w3) + b3)
L3 = tf.nn.dropout(L3, rate = keep_prob)


w4 = tf.get_variable("W4", shape=[512, 216], 
                     initializer=tf.contrib.layers.xavier_initializer())   
b4 = tf.Variable(tf.zeros([216]))
L4 = tf.nn.selu(tf.matmul(L3, w4) + b4)
L4 = tf.nn.dropout(L4, rate = keep_prob)



w5 = tf.get_variable("W5", shape=[216, 10], 
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.zeros([10]))
hypothesis = tf.nn.relu(tf.matmul(L4, w5) + b5)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=y))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


with  tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs) :  # 15
        
        avg_cost = 0
        
        for i in range(total_batch) :   # 600
         
            start  = i * batch_size
            end = start + batch_size

            batch_xs, batch_ys = x_train[start : end], y_train[start : end]
            
            
            feed_dict = {x:batch_xs, y: batch_ys, keep_prob : 0.7}   
            c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)   
            avg_cost += c/ total_batch 
            
        print(f"Epoch: {(epoch + 1):04d}, Cost: {avg_cost:.9f}")
       


        prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))            

    print("Accuracy:",sess.run(accuracy, feed_dict={x:x_test, y:y_test, keep_prob : 1})) 


