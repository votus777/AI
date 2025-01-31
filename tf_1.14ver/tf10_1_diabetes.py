
from sklearn.datasets import load_diabetes 

import tensorflow as tf


diabetes = load_diabetes()
x_data, y_data = diabetes.data, diabetes.target


x = tf.placeholder(tf.float32, shape=[442,10])
y = tf.placeholder(tf.float32, shape=[442,])

w = tf.Variable(tf.random_normal([10, 1]), name = 'Weight')
b = tf.Variable(tf.zeros([442]), name = 'bias')


hypothesis = tf.sigmoid(tf.matmul(x,w) + b)  

cost = -tf.reduce_mean( y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)

train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype =tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype =tf.float32))


with tf.Session() as sess :

    sess.run(tf.global_variables_initializer())

    for step in range(2001) :
    
        cost_val,  _ = sess.run([cost, train], 
                                feed_dict ={x : x_data, y : y_data})   
    
        if step % 500 ==0 :
            print(step, 'cost : ', cost_val )              

            h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict = {x:x_data, y:y_data})
    
print("\nHypotheis : ", h, "\n Correct (y) : ", c , "\n Accuracy : ", a  )    
