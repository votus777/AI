
from urllib.parse import quote_plus, urlencode

# from urllib2 import Request, urlopen
# from urllib import   urlencode
from urllib.request import urlopen, Request

url = 'http://data.ex.co.kr/openapi/trafficapi/sectionTrafficRouteDirection'
queryParams = '?' + urlencode({ quote_plus('ServiceKey') : '9934372478', 
                   collectDate('collectDate') : '20200401', quote_plus('routeId') : 'DJB30300052' })

request = Request(url + queryParams)
request.get_method = lambda: 'GET'
response_body = urlopen(request).read()

print (response_body)