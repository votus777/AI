

# Python 샘플 코드 #

from urllib.parse import quote_plus, urlencode

# from urllib2 import Request, urlopen
# from urllib import   urlencode
from urllib.request import urlopen, Request

url = 'http://openapi.tago.go.kr/openapi/service/BusLcInfoInqireService/getRouteAcctoBusLcList'
queryParams = '?' + urlencode({ quote_plus('ServiceKey') : '3ndGf0faDVz91pcADU%2FbcI9324flmmj8ODyOrcSyvVHpqFuNLxehtJ%2FVx%2BD5Q6ZP%2F4YHN7kAX8Ni7s99F7%2F0aw%3D%3D', 
                   quote_plus('cityCode') : '25', quote_plus('routeId') : 'DJB30300052' })

request = Request(url + queryParams)
request.get_method = lambda: 'GET'
response_body = urlopen(request).read()

print (response_body)