
import pyodbc as pyo

server = '127.0.0.1'
database = 'bitdb'
username = 'bit2'
password = '1234'

conn = pyo.connect('DRIVER={ODBC Driver 17 for SQL Server}; SERVER='+server+
                   '; PORT= 1433; DATABASE=' +database+
                   '; UID=' +username+
                   '; PWD=' +password)

cursor = conn.cursor()

tsql = 'SELECT * FROM iris_2;'

# with cursor.execute(tsql):
#     row = cursor.fetchone()
    
#     while row : 
#         print(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + " " +
#               str(row[3]) + " " + str(row[4]))
#         row = cursor.fetchone()
        
from flask import Flask, render_template

app = Flask(__name__)       

@app.route('/sqltable')
def sqltable() :
    cursor.execute(tsql)
    return render_template('myweb.html', rows =  cursor.fetchall())
        
if __name__ == '__main__' :
    app.run(host = '127.0.0.1', port = 5000, debug = False)
        
        
# 브라우저에  http://127.0.0.1:5000/sqltable 입력 

conn.close()

