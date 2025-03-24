# 大模型部分算法
## 环境
`pip install -r requirements.txt`

## 调用方法
使用`gunicorn -w 4 -b 0.0.0.0:2400 main_news_api:app` 启动服务，默认为2400端口
1. 新闻分类接口调用：response = requests.get("http://localhost:2400/trigger/news")，或者`python test-title.py`内部循环调用十次
2. 新闻翻译与摘要接口调用：response = requests.get("http://localhost:2400/trigger/content")，或者`python test-content.py`循环调用十次

## 特殊补充
1. embedding不存在时，使用`python add_embedding_time.py`进行特征添加，会直接刷数据库添加向量，避免因为程序重启而导致变量漏加
2. embedding仅导入最近48小时的新闻，每1小时更新一次

## 设计思路
1. 基于[聚合API平台](https://api.bltcy.ai/)提供所有大模型服务，目前为内测账号，后期需要替换。有时服务不稳定，需要考虑其他更好接口
2. embedding在标题阶段调用并使用faiss进行去重，阈值为0.8
3. 在翻译与摘要阶段使用Doubao模型，在分类与选择新闻阶段使用GPT-4o增加稳定性
4. 使用trigger_content_processing进行定时任务
5. news接口一次处理20条新闻分类，content接口一次处理10条新闻翻译与摘要
6. 建议定期使用add_embedding进行特征添加，会直接刷数据库添加向量，避免因为程序重启而导致变量漏加