# -*- coding: utf8 -*-
import requests
import sys, time, json

url = 'http://localhost:2206/rast'

dialog_list = ['拉娜·德雷这个人你听说过吗？', '她是很有名气的一位女歌手。', '是啊，我特别喜欢她。']
response = requests.post(url, data={'dialog_turns': dialog_list})
obj = response.json()
print(obj)
