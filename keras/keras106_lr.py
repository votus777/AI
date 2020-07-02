
weight = 0.5
input = 0.5
goal_prediction = 0.8 

lr = 0.001 

for iteration in range(1101) :                          # 1100번 돌린다 
    prediction = input*weight                           # 초기 input을 넣은 예측값 0.5 * 0.5 = 0.25 
    error = (prediction - goal_prediction)**2           # goal_prediction 과의 차이의 제곱   (0.25 - 0.8) **2  = 0.3025 = error
    
    print("Error : ", str(error) + "\tPrediction : ", str(prediction))

    up_prediction = input * (weight + lr)                   
    up_error = (goal_prediction - up_prediction) ** 2
    
    down_prediction = input * (weight - lr)
    down_error = (goal_prediction - down_prediction) ** 2  # up & down  둘 다 계산해보고 
    
    if(down_error < up_error) :                             # 가중치가 lr만큼 움직일 방향을 정한다 
        weight = weight - lr 
    if(down_error > up_error) :
        weight = weight + lr 
        
            