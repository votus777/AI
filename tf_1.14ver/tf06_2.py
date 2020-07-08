
# tf.06.01py 카피
# lr 수정해서 연습 
# 0.01 -> 0.1 / 0.001/ 1
# epoch 2000번 보다 작게 

import tensorflow as tf

tf.set_random_seed(777)

# x_train = [1,2,3]
# y_train = [3,5,7]

x_train = tf.placeholder(tf.float32, shape = [None])  # placeholder & feed_dict 사용 
y_train = tf.placeholder(tf.float32, shape = [None])


W = tf.Variable(tf.random.normal([1]), name = 'weight')  
b = tf.Variable(tf.zeros([1]), name = 'bias')   # 가중치 초기값 = 0 



hypothesis = x_train * W + b 

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) 

train = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)


with tf.Session() as sess :   
    sess.run(tf.global_variables_initializer())  

    for step in range(500) :
        _, cost_val, W_val, b_val = sess.run([train, cost, W , b], feed_dict={x_train : [1,2,3], y_train : [3,5,7]})   
        
        if step % 20 == 0:  
            
            print(step, cost_val, W_val,b_val)


    # predict 구하기 

    print('예측값 : ', sess.run(hypothesis, feed_dict = {x_train :[5,6]}))    

    