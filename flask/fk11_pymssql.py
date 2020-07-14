
# MS SQL에 있는 데이터를 가져오자 

import pymssql as ms

conn = ms.connect(server = '127.0.0.1', user = 'bit2', password = '1234',
           database = 'bitdb') 

cursor = conn.cursor()  # 커서 생성 

cursor.execute('SELECT * FROM iris_2;')

row = cursor.fetchone()     # 행지정, 커서에 있는 한 줄을 가져올 것이다. 

while row : 
    # print("첫 철럼 : ",row[0], "두번쨰 컬럼 : " , row[1])
    print("첫 철럼 : %s 두번쨰 컬럼 : %s"  %(row[0], row[1]))
    
    row = cursor.fetchone()
    
conn.close()

