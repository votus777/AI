

# Python 샘플 코드 #

from urllib.parse import quote_plus, urlencode

# from urllib2 import Request, urlopen
# from urllib import   urlencode
from urllib.request import urlopen, Request

url = 'http://openapi.tago.go.kr/openapi/service/BusLcInfoInqireService/getRouteAcctoBusLcList'
queryParams = '?' + urlencode({ quote_plus('ServiceKey') : 'Jcom6CPLQ2mAVjjg3LsZsHGqoj2hhTvJ6CpzCbY686VJUs1KYSuCH8ZLFefLGugy%2F4CjeUDKcv9vMubiW8RE3Q%3D%3D', 
                   quote_plus('cityCode') : '25', quote_plus('routeId') : 'DJB30300052' })

request = Request(url + queryParams)
request.get_method = lambda: 'GET'
response_body = urlopen(request).read()

print (response_body)