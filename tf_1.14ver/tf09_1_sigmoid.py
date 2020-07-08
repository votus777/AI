import tensorflow as tf

tf.set_random_seed(777)

x_data = [[1, 2 ],
          [2, 3 ],
          [3, 1 ],
          [4, 3 ],
          [5, 3 ],
          [6, 2 ]]


y_data = [[0], 
          [0], 
          [0], 
          [1],
          [1],
          [1]]

x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random_normal([2, 1]), name = 'Weight')
b = tf.Variable(tf.zeros([1]), name = 'bias')


hypothesis = tf.sigmoid(tf.matmul(x,w) + b)  

# cost = tf.reduce_mean(tf.square(hypothesis - y ))

cost = -tf.reduce_mean( y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 2e-4)

train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype =tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype =tf.float32))


with tf.Session() as sess :

    sess.run(tf.global_variables_initializer())

    for step in range(2001) :
    
        cost_val,  _ = sess.run([cost, train], 
                                feed_dict ={x : x_data, y : y_data})   
    
        if step % 100 ==0 :
            print(step, 'cost : ', cost_val )              

        h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict = {x:x_data, y:y_data})
    
    print("\nHypotheis : ", h, "\n Correct (y) : ", c , "\n Accuracy : ", a  )    
