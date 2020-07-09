import tensorflow as tf
import numpy as np

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]
         

y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]
          
          
x = tf.placeholder(tf.float32, shape=[None,4])
y = tf.placeholder(tf.float32, shape=[None,3])         



w = tf.Variable(tf.random_normal([4,3]), name = 'Weight')
b = tf.Variable(tf.zeros([1,3], name = 'bias')) #  [3,], [3] 다 된다  
         
hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)  # 모든 output의 합이 1이 될 수 있도록 (softmax)
                                     # tf.nn  : Wrappers for primitive Neural Net (NN) Operations.

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1)) # categorical crossentropy 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)



with tf.Session() as sess :

    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, loss_val = sess.run([optimizer,loss],
                               feed_dict = {x : x_data, y: y_data})

        if step % 200 == 0 :
            print(step, loss_val)
            
            
    #---여기까지 for문이 끝나고 최적의 w와 b가 구해져있는 상태

    predict_1 = sess.run(hypothesis, feed_dict = {x:[[1, 3, 4, 3]]})  
    predict_2 = sess.run(hypothesis, feed_dict = {x:[[1, 11, 7, 9]]})  
    predict_3 = sess.run(hypothesis, feed_dict = {x:[[11, 33, 4, 13]]})  
    

    print(predict_1, sess.run(tf.argmax(predict_1,1)))  # arg(predict,1) -> 이 predict의, 행(1) 중 가장 큰 값 return 
    print(predict_2, sess.run(tf.argmax(predict_2,1)))    
    print(predict_3, sess.run(tf.argmax(predict_3,1)))    
    
    
    # predict_1, predict_2, predict_3 넣어서 완성하자 

    dict_all = [[1, 3, 4, 3], [1, 11, 7, 9], [11, 33, 4, 13]]
    
    all = sess.run(hypothesis, feed_dict = { x : dict_all})
    print(all, sess.run(tf.argmax(all,1)))