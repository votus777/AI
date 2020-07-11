import tensorflow as tf
import numpy as np

from keras.datasets import mnist

# 데이터 

(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape) #(60000, 28, 28)
print(y_train.shape) #(60000,)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2]).astype('float32')/255
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2]).astype('float32')/255

# 변수 설정 
learning_rate = 0.001 
training_epochs = 5000
batch_size = 100 
total_batch = int(len(x_train)/ batch_size) # 60000 / 100


x = tf.placeholder(tf.float32, shape=[None,28*28])
y = tf.placeholder(tf.float32, shape=[None,10])
keep_prob = tf.placeholder(tf.float32)

# w1 = tf.Variable(tf.random.normal([784,512]), names='weight')  
w1 = tf.get_variable("w1", shape=[784,256], 
                     initializer=tf.contrib.layers.xavier_initializer())    # 기존에 쓰던 Variable보다 업그레이드 된 녀석, 초기값이 없어도 자기가 알아서 생성  
b1 = tf.Variable(tf.zeros([256]))
L1 = tf.nn.selu(tf.matmul(x,w1) + b1) 
L1 = tf.nn.dropout(L1, rate=  keep_prob)

# print(w1) :  shape=(784, 256) 
# print(b1) :  shape=(256,) 
# print(L1) :  shape=(?, 256)
# print(L1) :  shape=(?, 256)


w2 = tf.get_variable("w2", shape=[256,256], 
                     initializer=tf.contrib.layers.xavier_initializer())    # 표준 정규 분포를 입력 개수의 표준 편차로 나눔 -> w = np.random.randn(n_input, n_output) / sqrt(n_input)
b2 = tf.Variable(tf.zeros([256]))
L2 = tf.nn.selu(tf.matmul(L1,w2) + b2) 
L2 = tf.nn.dropout(L2, rate = keep_prob)


w3 = tf.get_variable("W3", shape=[256, 256], 
                     initializer=tf.contrib.layers.xavier_initializer())    # xavier 초기화 할 때는 relu 같이 못쓴다 (0으로 수렴), 대신 relu는 He 초기화 w = np.random.randn(n_input, n_output) / sqrt(n_input/2)
b3 = tf.Variable(tf.zeros([256]))
L3 = tf.nn.relu(tf.matmul(L2, w3) + b3)
L3 = tf.nn.dropout(L3, rate = keep_prob)


w4 = tf.get_variable("W4", shape=[256, 10], 
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.zeros([10]))
hypothesis = tf.nn.relu(tf.matmul(L3, w4) + b4)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=y))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with  tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs) :  # 15
        
        avg_cost = 0
        
        for i in range(total_batch) :   # 600
            
            ####################################################
            # batch_xs, batch_ys =  x_train([batch_size])  이 부분 구현하기
            # batch_xs, batch_ys = x_train[i*batch_size:i*batch_size+batch_size], y_train[i*batch_size:i*batch_size+batch_size]
         
            start  = i * batch_size
            end = start + batch_size

            batch_xs, batch_ys = x_train[start : end], y_train[start : end]
            
            
            feed_dict = {x:batch_xs, y: batch_ys, keep_prob : 0.7}   # 여기서 dropout 반대임, 0.7만을 남기겠다 
            c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)   
            avg_cost += c/ total_batch 
            
        print(f"Epoch: {(epoch + 1):04d}, Cost: {avg_cost:.9f}")
       


        prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))            

        print("Accuracy:",sess.run(accuracy, feed_dict={x:x_test, y:y_test, keep_prob : 1})) # Acc 출력 


