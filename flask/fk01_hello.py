
# cmd

# ping naver.com
# 125.209.222.142 

# 동적 IP VS 고정 IP

# ipconfig 



from flask import Flask

app = Flask(__name__)

@app.route('gema')   # / 치면 ~가 나오도록 
def hello333():
    return "<h1</Hello gema-camp</h1>"

app.route('/bit')
def hellow334() :
    return "<h1>Hello<h1>" 

app.route('/bit/bitcamp')
def hellow335() :
    return "<h1>Hello< bitcamp world<h1>" 

if __name__=='__main__' :
    app.run(host='127.0.0.1',port =8888, debug = True)
    
    