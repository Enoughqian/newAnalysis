import os

# LLM配置
LLM_CONFIG = {
    'API_KEY': '82455d5f-5d05-4fa3-988d-f5b4c9d173d9',
    "BASE_URL": "https://ark.cn-beijing.volces.com/api/v3",
    'MODEL_NAME': 'doubao-seed-2-0-pro-260215',
    'REASONING_EFFORT': 'medium',
    'COSTS': {
        "doubao-seed-2-0-pro-260215": {"prompt": 0.0032 / 1000, "completion": 0.016 / 1000},
        "doubao-seed-2-0-lite-260215": {"prompt": 0.0006 / 1000, "completion": 0.0036 / 1000},
        "doubao-embedding-large-text-240915": {"prompt": 0.0007},
    }
}

# DOUBAO LLM配置（翻译与摘要）
DOUBAO_CONFIG = {
    'API_KEY': '82455d5f-5d05-4fa3-988d-f5b4c9d173d9',
    'BASE_URL': 'https://ark.cn-beijing.volces.com/api/v3',
    'MODEL_NAME': "doubao-seed-2-0-lite-260215",
    'REASONING_EFFORT': 'low',
}

# Embedding配置
EMBEDDING_CONFIG = {
    'API_KEY': '82455d5f-5d05-4fa3-988d-f5b4c9d173d9',
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