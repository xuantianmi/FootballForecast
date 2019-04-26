#!/usr/bin/env python3
# coding=utf-8
"""
python=3.7.0
"""


class HttpStatusError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)