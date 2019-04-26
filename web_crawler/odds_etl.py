#!/usr/bin/env python3
# coding=utf-8
"""
python=3.7.0
"""

import numpy as np
import csv
import os
import sys
import xlrd

import tools

from custom_error import HttpStatusError


def parse_excel_data4file(odds_file):
    # 打开文件
    workbook = xlrd.open_workbook(odds_file)

    # 根据sheet索引或者名称获取sheet内容, sheet索引从0开始
    sheet1 = workbook.sheet_by_index(0)

    # 获取第1列内容
    sample_col = sheet1.col_values(1)
    lines = len(sample_col)

    # 从第三行开始读取数据
    row_num = 2
    cache_row = []
    odds_matrix = [[]].pop(0)
    while row_num < lines:
        if row_num % 2 == 1:
            tmp_row = sheet1.row_values(row_num)
            # print("tmp_row len:{0}".format(len(tmp_row)))
            # 终赔+返还率+凯利指数
            extra_raw = tmp_row[7:17]

            cur_row = cache_row + extra_raw
            score = cur_row[5]
            # print(score)
            # 增加主队得分和客队得分两列
            score_list = ['', '', '']

            if score.find(':') == -1 or score.upper() == 'VS':
                cur_row[5] = ''
            else:
                tmp_list = score.split(':')
                tmp_list.append('0')
                # print("scores is {0}".format(tmp_list))
                score_list = tmp_list

                if score_list[0] > score_list[1]:
                    score_list[2] = '3'
                elif score_list[0] == score_list[1]:
                    score_list[2] = '1'
                else:
                    score_list[2] = '0'
            # print("score list is {0}".format(score_list))
            # print("cache_row len:{0}".format(len(cache_row)))
            # print("extra_raw len:{0}".format(len(extra_raw)))
            cur_row.insert(6, score_list[0])
            cur_row.insert(7, score_list[1])
            cur_row.insert(8, score_list[2])
            # print("row len:{0}".format(len(cur_row)))
            # print(cur_row)
            odds_matrix.append(cur_row)
        else:
            cache_row = sheet1.row_values(row_num)
            # 如果最后一场比赛的终赔为空，则对应行不会被加载，需做额外处理
            if row_num == lines - 1:
                odds_matrix.append(cur_row)

        row_num = row_num + 1

    return odds_matrix


def parse_data4day(odds_date, odds_multi_company):
    """
    ETL: 从Excel中读取赔率数据
    :param odds_date:
    :return:
    """
    odds_matrix = [[]].pop(0)

    for odds_company in odds_multi_company:
        tmp_file = "竞彩足球 - {0}期 - 欧洲指数 [{1}].xls".format(odds_date, odds_company)
        print("file info: {0}".format(tmp_file))
        tmp_matrix = parse_excel_data4file(tmp_file)

        # print("odds_matrix.{0}, tmp_matrix.{1}".format(len(odds_matrix), len(tmp_matrix)))
        if len(odds_matrix)>0 and len(odds_matrix) == len(tmp_matrix):
            tmp_matrix = np.array(tmp_matrix)
            tmp_matrix = tmp_matrix[:, 10:]
            odds_matrix = np.hstack((odds_matrix, tmp_matrix))
        else:
            odds_matrix = tmp_matrix

    # print("odds_matrix len:{0}".format(len(odds_matrix)))
    # print("odds_matrix:{0}".format(odds_matrix))
    # odds_matrix.insert(0, odds_date)
    # odds_matrix.insert(0, companyIDs[odds_company])
    return odds_matrix


def parse_odds_files(out_csv_file, odds_file_path, date_list, odds_multi_company):
    """
    ETL: 解析所有的Excel文件，并存储到CSV文件中
    :return: NONE
    """
    # 'a'追加，'w'重写
    out_file = open(out_csv_file, 'w', newline='', encoding='utf-8-sig')

    n = len(date_list)
    index = 0
    # 切换目录到Excel文件目录
    os.chdir(odds_file_path)
    while index < n:
        # 如下逻辑可以保证获取失败后，还可以进行重试（失败时，index不会递增）。
        try:
            date = date_list[index]
            print(date_list[index])
            # 解析赔率文件
            odds_csv_db = csv.writer(out_file, dialect='excel')
            cur_matrix = parse_data4day(date, odds_multi_company)
            print(cur_matrix)
            odds_csv_db.writerows(cur_matrix)

            index += 1
        except ValueError as err:
            print("Could not convert data to an integer. {0}".format(err))
        except (KeyError, RuntimeError, TypeError, NameError) as err:
            print("Error: {0}".format(err))
        except HttpStatusError as err:
            print("Http Error: {0}".format(err))
        except AttributeError as err:
            print("Http Error: {0}".format(err))
        except:
            print("Unexpected error:", sys.exc_info()[0])
        else:
            print("......")


def main():
    # date_list = odds_crawler.dayList
    date_list = tools.get_between_days("2015-01-01", "2019-04-25")
    odds_companys = ["平均赔率", "立博", "皇冠", "澳门"]
    csv_file = 'hello-odds-cids.csv'
    odds_file_path = "../odds-files/"
    # 解析所有文件
    parse_odds_files(csv_file, odds_file_path,  date_list, odds_companys)


if __name__ == "__main__":
    main()
