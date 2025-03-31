import os

# LLM配置
LLM_CONFIG = {
    # 'API_KEY': 'sk-V8RTUXs5GF4dQ5o4B0618fA095F9425986732423563115E',
    'API_KEY': '90f66b8f-7fa8-4276-a362-8d26055d1c32',
    # 'BASE_URL': 'https://api.bltcy.ai/v1',
    "BASE_URL": "https://ark.cn-beijing.volces.com/api/v3",
    'MODEL_NAME': 'deepseek-v3-250324',  # 以下不再可选: gpt-4o-2024-08-06, gpt-4o-2024-11-20
    'COSTS': {
        "gpt-4o-2024-11-20": {"prompt": 0.00625 / 1000, "completion": 0.025 / 1000},
        "gpt-4o-2024-08-06": {"prompt": 0.00625 / 1000, "completion": 0.025 / 1000},
        "deepseek-v3-250324": {"prompt": 0.002 / 1000, "completion": 0.008 / 1000},
        "doubao-1-5-pro-32k-250115": {"prompt": 0.0008 / 1000, "completion": 0.002 / 1000},
        "BAAI/bge-large-en-v1.5": {"prompt": 0.0002},
        "doubao-embedding-large-text-240915": {"prompt": 0.0007},
        "doubao-embedding-text-240715": {"prompt": 0.0005}
    }
}

# DOUBAO LLM配置
DOUBAO_CONFIG = {
    'API_KEY': '90f66b8f-7fa8-4276-a362-8d26055d1c32',
    'BASE_URL': 'https://ark.cn-beijing.volces.com/api/v3',
    'MODEL_NAME': "doubao-1-5-pro-32k-250115",  # 可选: doubao-1-5-pro-32k-250115, deepseek-r1-250120, deepseek-v3-250324
}

# Embedding配置
EMBEDDING_CONFIG = {
    'API_KEY': '90f66b8f-7fa8-4276-a362-8d26055d1c32',
    'BASE_URL': 'https://ark.cn-beijing.volces.com/api/v3',
    'MODEL_NAME': "doubao-embedding-large-text-240915", 
}

# 新闻服务API配置
NEWS_API_CONFIG = {
    'BASE_URL': 'http://152.32.218.226:9999/news_server/api',
    'RECALL_URL': 'http://152.32.218.226:9999/news_server/api/recallTask'
}

# 数据库配置
DB_CONFIG = {
    'host': '152.32.218.226',
    'user': 'work',
    'password': 'Sqe3u8N_VP',
    'database': 'spider_news'
}

# 向量搜索配置
VECTOR_SEARCH_CONFIG = {
    'DIMENSION': 4096, #1024,
    'THRESHOLD': 0.91,   # 相似度阈值越高，过滤掉的新闻越少，保留的新闻越多，建议为0.80-0.95
    'LOAD_HOURS': 48  # 导入最近48小时的数据
}
