# 이진 분류
from sklearn.datasets import load_breast_cancer
import tensorflow as tf


cancer = load_breast_cancer()
x_data, y_data = cancer.data, cancer.target


x = tf.placeholder(tf.float32, shape=[569,30])
y = tf.placeholder(tf.float32, shape=[569,])

w = tf.Variable(tf.zeros([30, 1]), name = 'Weight')
b = tf.Variable(tf.zeros([1]), name = 'bias')


hypothesis = tf.sigmoid(tf.matmul(x,w) + b)  

cost = -tf.reduce_mean( y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-6)

train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype =tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype =tf.float32))


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
