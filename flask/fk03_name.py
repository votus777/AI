from flask import Flask

app = Flask(__name__)

@app.route("/<name>")
def user(name):
    return '<h1>Hello, %s !!!</h1>' % name

@app.route("/user/<name>")
def user2(name):
    return '<h1>Hello, user/%s !!!</h1>' % name

if __name__ == '__main__' :
    app.run(host = '127.0.0.1', port = 5001 )

# 그냥 들어가면 404 not found error 

# http://127.0.0.1:5001/아무거나 처럼 / 뒤에 명시해주면 
# Hello, 아무거나 !!! 이렇ㄱ ㅔ나옴 