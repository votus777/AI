
# tensorflow 는 노드 연산을 한다 

# TensorFlow에서는 graph의 연산에게 직접 tensor 값을 줄 수 있는 'feed 메커니즘'도 제공합니다.

import tensorflow as tf


node1 = tf.constant(3.0 , tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)


sess = tf.Session() # 강의장 준비 ( tensor machine )

a = tf.placeholder(tf.float32)  # 강의장 예약 걸어놓기 ( 예약할 떄는 당연히 예상 인원수(input 여기서는 feed_dict) 필요 )
b = tf.placeholder(tf.float32)


adder_node = a + b  # 강의장 의자 배치

print(sess.run(adder_node, feed_dict={a :3 , b:4.5}))  # sess.run을 할 떄 집어넣는 input과 비슷한 개념 
print(sess.run(adder_node, feed_dict={a :[1,3] , b:[2,4]}))  # feed_dict = 들어오는 사람, 집어넣는 값, 해당 run()의 변수로만 사용

add_and_triple_node = adder_node * 3  # 강의장 의자 배치

print(sess.run(add_and_triple_node, feed_dict={a:3, b:4.5})) # sess.run = 강의 진행  

'''
7.5
[3. 7.]
22.5

결국 이 방식은 NN 방식의 node 그래프 연산 

그냥 node만 출력하면 자료형만 나오고 
이 node를 거친 출력값을 보려면 sess.run을 해야하고 

placeholder 또한 마찬가지 


'''

