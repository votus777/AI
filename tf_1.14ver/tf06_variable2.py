# hypothesis를 구하시오 
# H = Wx + b 


import tensorflow as tf
tf.set_random_seed(777)


x = [1,2,3]

W = tf.Variable(tf.Variable([0.3]), name = 'weight')  
b = tf.Variable(tf.zeros([1]), name = 'bias')   # 가중치 초기값 = 0 

hypothesis = x * W + b 

print(W) #  <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref> -> 자료형
print(hypothesis)


###################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(hypothesis) 
print('1 : ', hypothesis)
sess.close() 



# interactivesession도 있다
  
sess = tf.InteractiveSession()  
sess.run(tf.global_variables_initializer())
bbb = hypothesis.eval()
print('2 : ', hypothesis)  
sess.close()


# 그냥 session에서도 eval 쓸 수 있다 
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = hypothesis.eval(session = sess)
print('3 : ', hypothesis)
sess.close()


