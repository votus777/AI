import sqlite3 

conn = sqlite3.connect('test.db')

cursor = conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS supermarket(Itemno INTEGER, Category TEXT,
               FoodName TEXT, Company TEXT, Price INTEGER)""")

sql = "DELETE FROM supermarket"
cursor.execute(sql)  # 이게 없으면 실행시킬 때마다 중첩되서 나옴 

# 데이터 
sql = 'INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)'
cursor.execute(sql, (1,'과일', '자몽', '마트', 1500))

sql = 'INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)'
cursor.execute(sql, (2,'음료수', '망고주스', '편의점', 1000))

sql = 'INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)'
cursor.execute(sql, (33,'고기', '소고기', '하나로마트', 1000))

sql = 'INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)'
cursor.execute(sql, (4,'박카스', '약', '약국', 500))

sql = 'SELECT * FROM supermarket'
sql = 'SELECT Itemno, Category, FoodName, Company, Price from supermarket'
cursor.execute(sql)

rows = cursor.fetchall() 

for row in rows :
    print(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + " " +
          str(row[3]) + " " + str(row[4]))

conn.commit()
conn.close()