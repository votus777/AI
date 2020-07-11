import tensorflow as tf
import numpy as np

from keras.datasets import mnist

# 데이터 

(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape) #(60000, 28, 28)
print(y_train.shape) #(60000,)


x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2]).astype('float32')/255
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2]).astype('float32')/255

# y_train=y_train.reshape(-1,10)
# y_test=y_test.reshape(-1,10)


x = tf.placeholder(tf.float32, shape=[None,28*28])
y = tf.placeholder(tf.float32, shape=[None,10])

w1 = tf.Variable(tf.random_normal([28*28,100]),name="weight")
b1 = tf.Variable(tf.random_normal([100]),name="bias")
layer1 = tf.nn.elu(tf.matmul(x,w1)+b1)
layer1 = tf.nn.dropout(layer1, keep_prob =0.3)

# w2 = tf.Variable(tf.random_normal([50,25]),name="weight")
# b2 = tf.Variable(tf.random_normal([25]),name="bias")
# layer2 = tf.nn.elu(tf.matmul(layer1,w2)+b2)

w3 = tf.Variable(tf.random_normal([100,10]),name="weight")
b3 = tf.Variable(tf.random_normal([10]),name="bias")
hypothesis = tf.nn.softmax(tf.matmul(layer1,w3)+b3)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

predict = tf.equal(tf.argmax(hypothesis,1),tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    y_train = sess.run(tf.one_hot(y_train,10))
    y_test = sess.run(tf.one_hot(y_test,10))    

    for step in range(2000):
        
        _,loss_val,hypo_val=sess.run([optimizer,loss,hypothesis], feed_dict={x:x_train,y:y_train})
        
        if step % 100==0:
            print(loss_val)
            print(f"step:{step},loss_val:{loss_val}")
       

    print("Accuracy:",sess.run(accuracy,feed_dict={x:x_test,y:y_test}))

 
# step:1900,loss_val:1.5152500867843628
# Accuracy: 0.9361