# 3 + 4 + 5
# 4 - 3
# 3 * 4
# 4 / 2
 
import tensorflow as tf 

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0, tf.float32)
node3 = tf.constant(5.0, tf.float32)

node_sum = tf.add([node1,node2,node3],0)
node_sub = tf.subtract(node2, node1)   # sub에서 substract로 바뀜  
node_mul = tf.multiply(node1, node2)   # 역시 mul에서 multiply
node_div = tf.div(node2,2)

sess = tf.Session()
print('sum : ' ,sess.run(node_sum)) 
print('sub : ' ,sess.run(node_sub)) 
print('mul : ' ,sess.run(node_mul)) 
print('div : ' ,sess.run(node_div)) 

'''
sum :  [3. 4. 5.]
sub :  1.0
mul :  12.0
div :  2.0

'''