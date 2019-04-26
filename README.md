# 赛果预测模型

## 一维卷积胜平负预测
通过一维卷积神经网络模型，预测比赛的胜平负结果。

训练数据说明：
- features：各公司的开盘赔率、胜率、返还率和凯利指数
- labels：310

### 文件说明
- ./odds-files/ 获取的赔率文件（提供样例文件）
- ./web_crawler/odds_crawler.py 获取网站赔率文件（此逻辑简单但敏感，有兴趣请联系作者）
- ./web_crawler/odds_etl.py 解析赔率文件并保存成csv文件
- ./odds_forecast_cids.py 建立模型、培训、保存模型
- ./odds_forecast_predict.py 获得预测数据、加载模型及预测
- ./custom_error.py 自定义Error
- ./tools.py 通用工具