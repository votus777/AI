
import tensorflow as tf

tf.set_random_seed(777)

W = tf.Variable(tf.random.normal([1]), name = 'weight')  
b = tf.Variable(tf.zeros([1]), name = 'bias')   # 가중치 초기값 = 0 

print(W) #  <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref> -> 자료형

W = tf.Variable([0.3], tf.float32)

###################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(W) # [0.3]
print('1 : ', aaa)
sess.close() 
# 작은 소스에서는 상관없는데 큰 소스에서는 엉키게 될 수도 있다 



# interactivesession도 있다    run 대신에 eval 사용 , tf.InteractiveSession은 자동으로 터미널에 default session을 할당
sess = tf.InteractiveSession()  
sess.run(tf.global_variables_initializer())
bbb = W.eval()
print('2 : ', bbb)  #  [0.3]
sess.close()


# 그냥 session에서도 eval 쓸 수 있다 
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session = sess)
print('3 : ', ccc)
sess.close()


