import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

dataset = np.loadtxt('data\csv\data-tf.csv', delimiter=',', dtype= np.float32, encoding='UTF-8')

x_data = dataset[ : , 0 : -1]
y_data = dataset [ : , [-1]]

x_data = tf.placeholder(tf.float32, shape = [None, 3])
y_data = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name = 'Weight')
b = tf.Variable(tf.zeros([1]), name = 'bias')


hypothesis = tf.matmul(x_data,w) + b  # 행렬연산 matmul 

cost = tf.reduce_mean(tf.square(hypothesis - y ))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 2e-4)

train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001) :
    
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                                    feed_dict ={x : x_data, y : y_data})   
    
    if step % 10 ==0 :
        print(step, 'cost : ', cost_val , "\n", hy_val )              


sess.close()