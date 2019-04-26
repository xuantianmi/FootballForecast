#!/usr/bin/env python3
# coding=utf-8
"""
python=3.7.0
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder

# StratifiedShuffleSplit可以用来把数据集洗牌，并拆分成训练集和验证集
from sklearn.model_selection import StratifiedShuffleSplit
# 导入优化损失方法
from keras.optimizers import SGD

from keras.utils import np_utils
# 导入顺序模型，它由多个网络层线性堆叠
from keras.models import Sequential
# 导入可用于处理全连接层，激活函数，二维卷积，最大池化，压平数据包
from keras.layers import Dense, Activation, Convolution1D, Flatten, Dropout
from keras.models import model_from_json
from keras.models import model_from_yaml


'''
多家博彩的开盘赔率综合预测。对应odds_crawler.py and odds_etl.py
运用CNN1d，通过比赛赔率、胜率和凯利指数，预测比赛结果
- 根据fit 的结果  预期值和实际值  方差越小 拟合度越好 来决定各参数
- 很多参数 还有模型深度，要通过测试确定
- 评估提供模型的数据是否适合训练
REF: https://www.kaggle.com/alexanderlazarev/simple-keras-1d-cnn-features-split
'''


def load_csv_file(csv_file):
    # UTF-8编码格式csv文件数据读取
    # WARN: pd.read_csv默认从第二行取值，第一行默认为列名.
    raw_df = pd.read_csv(csv_file, header=None)  # 返回一个DataFrame的对象，这个是pandas的一个数据结构

    raw_df.columns = ["场次", "赛事", "round", "match_time",
                      "host", "score", "score1", "score2", "draw", "guest",
                      "010", "011", "012", "013", "014", "015", "016", "017", "018", "019",
                      "020", "021", "022", "023", "024", "025", "026", "027", "028", "029",
                      "110", "111", "112", "113", "114", "115", "116", "117", "118", "119",
                      "120", "121", "122", "123", "124", "125", "126", "127", "128", "129",
                      "210", "211", "212", "213", "214", "215", "216", "217", "218", "219",
                      "220", "221", "222", "223", "224", "225", "226", "227", "228", "229",
                      "310", "311", "312", "313", "314", "315", "316", "317", "318", "319",
                      "320", "321", "322", "323", "324", "325", "326", "327", "328", "329"]
    """
    编号含义：***-cid|初1/终2|顺序号，如020-平均赔率、终场情况、胜终赔
    - 0-平均赔率；1-立博, 2-皇冠, 3-澳门
    - 10-初胜赔, 11初平赔, 12初负赔, 13初胜率, 14初平率, 15初负率, 16初返还率, 17初凯利1, 18初凯利2, 19初凯利3,
    - 20-胜终赔, 21平终赔, 22负终赔, 23终胜率, 24终平率, 25终负率, 26终返还率, 27终凯利1, 28终凯利2, 29终凯利3,
    """

    return raw_df


def prepare(df):
    # 去除没有比赛结果的行记录
    df = df.dropna(axis=0)

    # 抽取初始信息（赔率/返还率/凯利指数）列作为训练数据的各属性值
    # 样本的特征值
    x = df[["010", "011", "012", "013", "014", "015", "016", "017", "018", "019",
            "110", "111", "112", "113", "114", "115", "116", "117", "118", "119",
            "210", "211", "212", "213", "214", "215", "216", "217", "218", "219",
            "310", "311", "312", "313", "314", "315", "316", "317", "318", "319"]].astype(float)
    # x = np.array(x)

    # 样本的目标值, "draw"列作为每行对应的标签label
    y1 = df["draw"].astype(int)
    y = np.array(y1)

    # 训练数据
    train_features = x
    train_labels = y

    # 用LabelEncoder为叶子的种类标签编码，labels对象是训练集上的标签列表
    label_encoder = LabelEncoder().fit(train_labels)
    train_labels = label_encoder.transform(train_labels)
    print("train_labels:{0}".format(train_labels))

    label_classes = list(label_encoder.classes_)
    print("label_classes:{0}".format(label_classes))
    print("train_features.shape:{0}".format(train_features.shape))

    return train_features, train_labels, label_classes


def prepare_etl(train, labels, classes):
    # label_classes:[0, 1, 3]
    # train, labels, t_features, t_labels, classes = prepare(raw_df)

    # 这里只是标准化训练集的特征值
    scaler = StandardScaler().fit(train.values)
    scaled_train = scaler.transform(train.values)

    # 把数据集拆分成训练集和测试集，测试集占10%
    sss = StratifiedShuffleSplit(test_size=0.1, random_state=23)
    for train_index, valid_index in sss.split(scaled_train, labels):
        X_train, X_valid = scaled_train[train_index], scaled_train[valid_index]
        y_train, y_valid = labels[train_index], labels[valid_index]

    print("x_train shape:{0}".format(X_train.shape))

    # 每个输入通道的大小是nb_features位
    nb_features = 8
    # 通道数, nb_features*channels=训练样本特征（即列）的数量
    channels = 5
    nb_class = len(classes)

    print("len(X_train):{0}".format(len(X_train)))
    #  把输入数据集reshape成keras喜欢的格式：（样本数，通道大小，通道数）
    X_train_r = np.zeros((len(X_train), nb_features, channels))
    print("X_train_r's shape:{0}".format(X_train_r.shape))
    print("X_valid shape:{0}".format(X_valid.shape))

    # 先把所有元素初始化成0之后，再把刚才的数据集中的数据赋值过来
    # (Merlin)将二维数组的不同列复制到三维数组的不同层
    X_train_r[:, :, 0] = X_train[:, :nb_features]
    X_train_r[:, :, 1] = X_train[:, nb_features:nb_features * 2]
    X_train_r[:, :, 2] = X_train[:, nb_features * 2:nb_features * 3]
    X_train_r[:, :, 3] = X_train[:, nb_features * 3:nb_features * 4]
    X_train_r[:, :, 4] = X_train[:, nb_features * 4:]

    # 验证集也要reshape一下
    X_valid_r = np.zeros((len(X_valid), nb_features, channels))
    X_valid_r[:, :, 0] = X_valid[:, :nb_features]
    X_valid_r[:, :, 1] = X_valid[:, nb_features:nb_features * 2]
    X_valid_r[:, :, 2] = X_valid[:, nb_features * 2:nb_features * 3]
    X_valid_r[:, :, 3] = X_valid[:, nb_features * 3:nb_features * 4]
    X_valid_r[:, :, 4] = X_valid[:, nb_features * 4:]

    y_train = np_utils.to_categorical(y_train, nb_class)
    y_valid = np_utils.to_categorical(y_valid, nb_class)

    return nb_class, nb_features, channels, X_train_r, y_train, X_valid_r, y_valid


def build_new_model(nb_features, channels, nb_class):
    # 下面是Keras的一维卷积实现，原作者尝试过多加一些卷积层，
    # 结果并不能提高准确率，可能是因为其单个通道的信息本来就太少，深度太深的网络本来就不适合
    new_model = Sequential()

    # 一维卷积层用了256个卷积核，输入是nb_features*channels的格式
    # 此处要注意，一维卷积指的是卷积核是1维的，而不是卷积的输入是1维的，1维指的是卷积方式
    new_model.add(Convolution1D(nb_filter=256, filter_length=1, input_shape=(nb_features, channels)))
    # print("Convolution1D model.output_shape:{0}".format(new_model.output_shape))
    #
    new_model.add(Activation('relu'))
    # print("Activation model.output_shape:{0}".format(new_model.output_shape))
    # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
    new_model.add(Flatten())
    # print("Flatten model.output_shape:{0}".format(new_model.output_shape))
    # Dropout是一种在训练过程中随机忽略神经元的技术。他们随机“Dropout”, 意味着它们对下游神经元激活的贡献在正向通过时暂时消除，并且任何权重更新都不会发生于这些神经元。
    # 正则化，防止过拟合，随机过滤掉40%->20%的神经元
    new_model.add(Dropout(0.2))
    # print("Dropout model.output_shape:{0}".format(new_model.output_shape))
    # 常用的的全连接层+激活函数relu
    new_model.add(Dense(2048, activation='relu'))
    # print("Dense 2048 model.output_shape:{0}".format(new_model.output_shape))
    new_model.add(Dense(1024, activation='relu'))
    # print("Dense 1024 model.output_shape:{0}".format(new_model.output_shape))
    new_model.add(Dense(nb_class))
    # print("Dense nb_class model.output_shape:{0}".format(new_model.output_shape))

    # softmax经常用来做多类分类问题
    new_model.add(Activation('softmax'))

    print("nb_class:{0}".format(nb_class))

    # Compile model
    sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
    new_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    new_model.summary()
    return new_model


def train_model(model, X_train_r, y_train, X_valid_r, y_valid):
    # nb_epoch = 15->6
    nb_epoch = 15
    model.fit(X_train_r, y_train, nb_epoch=nb_epoch, validation_data=(X_valid_r, y_valid), batch_size=16)

    # evaluate the model
    scores = model.evaluate(X_valid_r, y_valid, verbose=0)
    print("model evaluate: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


def save_model2files(model, model_json_file, model_yaml_file, model_weight_file):
    # 保存训练好的模型: model architecture and model weights.
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_json_file, "w") as json_file:
        json_file.write(model_json)
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(model_yaml_file, "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(model_weight_file)
    print("Saved model to disk")


def load_json_model(model_json_file, model_weight_file):
    # load json and create model
    json_file4load = open(model_json_file, 'r')
    loaded_model_json = json_file4load.read()
    json_file4load.close()
    new_model = model_from_json(loaded_model_json)
    # load weights into new model
    new_model.load_weights(model_weight_file)
    print("Loaded model from disk")
    return new_model


def load_yaml_model(model_yaml_file, model_weight_file):
    # load YAML and create model
    yaml_file4load = open(model_yaml_file, 'r')
    loaded_model_yaml = yaml_file4load.read()
    yaml_file4load.close()
    new_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    new_model.load_weights(model_weight_file)
    print("Loaded model from disk")
    return new_model


def evaluate_model(cur_model, x, y):
    # evaluate loaded model on test data
    cur_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = cur_model.evaluate(x, y, verbose=0)
    print("%s: %.2f%%" % (cur_model.metrics_names[1], score[1] * 100))


def prepare_evaluate(sample_x, sample_y, nb_features, channels, nb_class):
    # 只是标准化训练集的特征值
    scaler = StandardScaler().fit(sample_x)
    scaled_sample = scaler.transform(sample_x)

    x_sample_r = np.zeros((len(scaled_sample), nb_features, channels))
    x_sample_r[:, :, 0] = scaled_sample[:, :nb_features]
    x_sample_r[:, :, 1] = scaled_sample[:, nb_features:nb_features * 2]
    x_sample_r[:, :, 2] = scaled_sample[:, nb_features * 2:nb_features * 3]
    x_sample_r[:, :, 3] = scaled_sample[:, nb_features * 3:nb_features * 4]
    x_sample_r[:, :, 4] = scaled_sample[:, nb_features * 4:]

    if sample_y is None:
        sample_y_r = None
    else:
        # 用LabelEncoder为叶子的种类标签编码，labels对象是训练集上的标签列表
        label_encoder = LabelEncoder().fit(sample_y)
        sample_y = label_encoder.transform(sample_y)
        # 将labels转为one-hote模式
        sample_y_r = np_utils.to_categorical(sample_y, nb_class)

    return x_sample_r, sample_y_r


def main():
    csv_file = './web_crawler/hello-odds-cids.csv'
    file_head = "odds_forecast_cids"
    model_json_file = file_head + ".json"
    model_weight_file = file_head + ".h5"
    model_yaml_file = file_head + ".yaml"

    raw_df = load_csv_file(csv_file)
    train_features, train_labels, label_classes = prepare(raw_df)
    nb_class, nb_features, channels, x_train_r, y_train, x_valid_r, y_valid \
        = prepare_etl(train_features, train_labels, label_classes)

    """
    # 生成模型并进行训练，并保存模型和权重参数
    created_model = build_new_model(nb_features, channels, nb_class)
    train_model(created_model, x_train_r, y_train, x_valid_r, y_valid)
    save_model2files(created_model, model_json_file, model_yaml_file, model_weight_file)
    """

    # 通过评估数据进行模型评估
    print("x_valid_r:{0}".format(x_valid_r))
    # test_x, test_y = prepare_evaluate(x_valid_r, y_valid, nb_features, channels, nb_class)
    test_x = x_valid_r
    test_y = y_valid

    # loaded_model = load_json_model()
    loaded_model = load_yaml_model(model_yaml_file, model_weight_file)
    evaluate_model(loaded_model, test_x, test_y)
    # 结果acc: 70.18%

    """
    # make a prediction
    y_new = loaded_model.predict_classes(test_x)
    # show the inputs and predicted outputs
    for i in range(len(x_new)):
        print("X=%s, Predicted=%s" % (i, y_new[i]))
    """


if __name__ == "__main__":
    main()
