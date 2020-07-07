
# 좌측 하단 버전 변경을 해주자

import tensorflow as tf
print(tf.__version__)  # 1.14.0

#constant = 상수 
hello = tf.constant('Hello, World')

print(hello)  # Tensor("Const:0", shape=(), dtype=string)



sess = tf.Session()
print(sess.run(hello)) # 'Hello, World'

# 텐서플로를 통과할 떄는 텐서플로 형식으로만 보여주기 때문에 ( 텐서플로 동작 원리 참조)
# 그래서  항상  Session으로 걸러서 봐야한다. 

