# !/usr/bin/python
# -*- coding:utf-8 -*-
from requests.models import PreparedRequest
import tornado
import tornado.httpclient as httpclient

asyncHTTPClient = httpclient.AsyncHTTPClient()
c = 0
file = 'test.jpg'
files = [
    ('image', ('%d' % c, open(file, 'rb'), 'image/jpg')),
]
data = {
    'filename': 'test',
}


def on_response(response):
    pass


if __name__ == "__main__":
    IP = '106.14.70.249'
    url = 'http://%s:9331/api/image_save' % IP
    prepared_request = PreparedRequest()
    prepared_request.method = 'post'
    prepared_request.headers = {}
    prepared_request.prepare_body(data, files)
    request = tornado.httpclient.HTTPRequest(url=url, headers=prepared_request.headers,
                                             method='POST', body=prepared_request.body)
    asyncHTTPClient.fetch(request, on_response)

    tornado.ioloop.IOLoop.instance().start()  # start the tornado ioloop to
