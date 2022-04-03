# !/usr/bin/python
# -*- coding:utf-8 -*-
import time


def timestamp2date(timestamp):
    ms = timestamp % 1000
    timeArray = time.localtime(timestamp / 1000)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime + ' ' + str(ms)


def date2timestamp(s):
    s1 = s[:-4]
    s2 = s[-4:]
    timeArray = time.strptime(s1, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp + float(s2) / 1000


if __name__ == "__main__":
    print(timestamp2date(int(time.time() * 1000)))
    print('*************************************')
    date = '2020-05-18 20:01:03 863'
    print(date2timestamp(date))
