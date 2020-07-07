
import tensorflow as tf

tf.set_random_seed(777)

x_train = [1,2,3]
y_train = [3,5,7]


W = tf.Variable(tf.random_normal([1]), name = 'weight')   #[1] = 1 차원  
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# tf.random.normal = 정규분포 난수에서의 random output 


hypothesis = x_train * W + b 

cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # loss(=cost) ->  mse

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess :    # sess 범주 안에 있는 것들, 원래 session을 열면 다시 닫아야 하지만 (  sess.close() ) , 이렇게 with 문을 사용하면 안닫아도 된다 메소드가  __enter__ 및 __exit__으로 이루어져있기 때문 
    sess.run(tf.global_variables_initializer())    # variable 변수는 항상 초기화를 시켜주어야 한다, 변수가 들어갈 메모리 할당       # print(sess.run(W))   # [[{{node _retval_weight_0_0}}]] 안그럼 이런 에러 뜬다  ->   [2.2086694]
                                                    # 참고로 tf.variable은 default로 'trainable=True' 라서 그래프 연산할 때 자동으로 학습된다  
    for step in range(2001) :
        _, cost_val, W_val, b_val = sess.run([train, cost, W , b])  # _,   -> 연산은 하지만 결과값 출력을 무시 하겠다 
        
        if step % 20 == 0:  # 20번마다 한번씩
            
            print(step, cost_val, W_val,b_val)




'''

step   cost_val       W_val        b_val
1980 2.6721513e-05 [2.0059893] [0.98638463]
2000 2.4268587e-05 [2.005708] [0.98702455]

y = Wx + b  에서  y = 2x+b 가 되었다 


'''

