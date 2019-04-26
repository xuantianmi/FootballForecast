#!/usr/bin/env python3
# coding=utf-8
"""
python=3.7.0
"""

from sklearn.preprocessing import LabelEncoder

from web_crawler import odds_crawler
from web_crawler import odds_etl
import odds_forecast_cids as predictor
import tools


'''
通过预测模型进行预测。based on odds_forecast_cids.py
ref: *https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/
'''


def fetch_from_500(date_list, odds_multi_company):
    odds_crawler.dayList = date_list

    for odds_company in odds_multi_company:
        odds_crawler.cid = odds_crawler.companyIDs[odds_company]
        odds_crawler.fetch_odds_files()


def predict(csv_file):
    """
    读取赔率的cvs文件，并进行预测
    :param csv_file:
    :return:
    """
    raw_df = predictor.load_csv_file(csv_file)
    test_features = prepare(raw_df)

    test_x, test_y = predictor.prepare_evaluate(test_features.values, None, 8, 5, 3)

    file_head = "odds_forecast_cids"
    model_weight_file = file_head + ".h5"
    model_yaml_file = file_head + ".yaml"
    loaded_model = predictor.load_yaml_model(model_yaml_file, model_weight_file)

    # 获取归一化的labes（及map）
    train_labels = ["3", "1", "0"]
    # 用LabelEncoder为叶子的种类标签编码，labels对象是训练集上的标签列表
    label_encoder = LabelEncoder().fit(train_labels)

    # make a prediction
    y_new = loaded_model.predict_classes(test_x)
    # show the inputs and predicted outputs
    for i in range(len(y_new)):
        result = label_encoder.inverse_transform([y_new[i]])
        print("平均赔率=%s, Predicted=%s" % (test_features.values[i, 0:3], result))


def prepare(df):
    # 抽取初始信息（赔率/返还率/凯利指数）列作为训练数据的各属性值
    # 样本的特征值
    print(df)
    features = df[["010", "011", "012", "013", "014", "015", "016", "017", "018", "019",
                   "110", "111", "112", "113", "114", "115", "116", "117", "118", "119",
                   "210", "211", "212", "213", "214", "215", "216", "217", "218", "219",
                   "310", "311", "312", "313", "314", "315", "316", "317", "318", "319"]].astype(float)
    # print(features)

    # 去除没有比赛赔率等信息的行记录
    features = features.dropna(axis=0)
    # features = np.array(x)

    return features


def main():
    csv_file = "predict_odds.csv"
    # 指定获取哪些公司的赔率文件
    odds_multi_company = ["平均赔率", "立博", "皇冠", "澳门"]
    # 指定获取哪些天的赔率文件
    data_list = tools.get_between_days("2019-04-24", "2019-04-24")
    # 获取赔率文件
    # fetch_from_500(data_list, csv_file, odds_multi_company)
    # 解析赔率文件
    # odds_etl.parse_odds_files(csv_file, ".", data_list, odds_multi_company)
    # 加载模型，并进行预测
    predict(csv_file)


def download():
    """
    从网站上获取赔率文件（日期+各赔率公司）
    :return:
    """
    odds_multi_company = ["平均赔率", "立博", "皇冠", "澳门"]
    data_list = tools.get_between_days("2019-03-30", "2019-04-25")
    fetch_from_500(data_list, odds_multi_company)


def parse_to_file():
    """
    解析获取的赔率文件，并保存到csv中
    注：可用于预测准备（无比赛结果），也可用于训练数据准备（有比赛结果）
    :return:
    """
    csv_file = "odds_forecast_data.csv"
    odds_multi_company = ["平均赔率", "立博", "皇冠", "澳门"]
    data_list = tools.get_between_days("2015-01-01", "2019-04-25")
    odds_etl.parse_odds_files(csv_file, "./web_crawler/", data_list, odds_multi_company)


if __name__ == "__main__":
    main()
    # download()
