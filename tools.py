#!/usr/bin/env python
# coding=utf-8
"""
python=3.5.2
"""

import datetime
import time


def get_between_day(begin_date):
    """
    获取从以往的指定时间到当前日期的日期列表
    :param begin_date:
    :return:
    """
    # begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(time.strftime('%Y-%m-%d', time.localtime(time.time())), "%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    return get_between_days(begin_date, end_date)


def get_between_days(begin_date, end_date):
    """
    获取从以往的指定时间1到以往的指定时间2的日期列表
    :param begin_date:
    :param end_date:
    :return:
    """
    date_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        begin_date += datetime.timedelta(days=1)
    return date_list


# print(get_between_days("2019-03-22", "2019-04-22"))
# print(get_between_day("2019-03-22"))
# print(get_between_day.__doc__)
