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

# 텐서플로 덧셈
# tf.add_n?
# 많은 양의 텐서를 한 번에 처리할 때 사용
# tf.add_n은 텐서들을 반드시 대괄호 안에 넣어줘야 한다(그렇지 않으면 TypeError 발생)
# and, tf.add(node1, variable2) 처럼 constant 텐서와 variable 텐서 간의 연산도 가능하다는 점 참고!

# 텐서플로 곱셈
# tf.multiply vs tf.matmul
# tf.multiply는 원소들의 곱 -> 우리가 알고 있는 행렬의 곱셈 방식이 아님
# tf.matmul은 행렬의 곱셈 . 함수를 통해 구현한다

# tf.divide vs tf.mod
# tf.divide : ~를 ~로 나누면?
# tf.mod : ~를 ~로 나눈 나머지?