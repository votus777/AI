import tensorflow as tf 

node1 = tf.constant(3.0 , tf.float32)
node2 = tf.constant(4.0 , tf.float32)
node3 = tf.add(node1, node2)

# contants는 변하지 않는 상수
# node는 라는 하나의 형태 

# 그것을 보려면 session에 넣어야 한다 

sess = tf.Session()



print('node1 : ', node1, 'node2 : ', node2)
print('node1 : ', sess.run(node1))
print('node2 : ', sess.run(node2))
print('node3 : ', sess.run(node3))

'''
node1 :  3.0
node2 :  4.0
node3 :  7.0
'''