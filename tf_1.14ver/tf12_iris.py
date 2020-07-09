
# 다중 분류
from sklearn.datasets import load_iris
import tensorflow as tf



iris = load_iris()
x_data, y_data = iris.data, iris.target

          
x = tf.placeholder(tf.float32, shape=[None,4])
y = tf.placeholder(tf.float32, shape=[None,3])    



w = tf.Variable(tf.random_normal([4,3]), name = 'Weight')
b = tf.Variable(tf.zeros([1,3 ], name = 'bias'))  


hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)  
                               
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1)) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)



with tf.Session() as sess :

    # One-Hot-Encoding 필요 
    y_data = sess.run(tf.one_hot(y_data,3))
    
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, loss_val = sess.run([optimizer,loss],
                               feed_dict = {x : x_data, y: y_data})

        if step % 200 == 0 :
            print(step, loss_val)

        
            