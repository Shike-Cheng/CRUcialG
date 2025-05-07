# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : logging.py
# Time       ：2023/6/7 10:18
# Author     ：Qixuan Yuan
# Description：
"""
import sys
import os
import sys
import datetime

def make_print_to_file(path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.path = os.path.join(path, filename)
            self.log = open(self.path, "a", encoding='utf8', )
            print("save:", os.path.join(self.path, filename))

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('day' + '%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60, '*'))


if __name__ == '__main__':
    make_print_to_file(path='./')

    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print("1234124")
    print("--")
    print(":;;;")
    print("")
    print("阿斯顿发11111111111111111")
    print("zzzzz")