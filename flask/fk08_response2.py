from flask import Flask, Response, make_response

app = Flask(__name__)

@app.route('/')

def response_test() :
    custom_response = Response('Custom Response',200,
                               {'Program' : 'Flask Web Application'})
    return make_response(custom_response)


@app.before_first_request
def before_first_request() : 
    print('앱이 가동되고 나서 첫번쨰 HTTP 요청에만 응답합니다.')  # route가 돌아가기 전에 이게 먼저 실행



if __name__ == '__main__' :
    app.run(host = '127.0.0.1', port = 5000, debug = False)  # port 따로 명시 안해도 5000에 할당됨(default)
    



    
    
