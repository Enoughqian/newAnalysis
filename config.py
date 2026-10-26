import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / '.env')

# LLM配置
LLM_CONFIG = {
    'API_KEY': os.environ['ARK_API_KEY'],
    "BASE_URL": "https://ark.cn-beijing.volces.com/api/v3",
    'MODEL_NAME': 'deepseek-v4-1-flash-economy-261017', #'doubao-seed-2-0-pro-260215',
    'REASONING_EFFORT': 'medium',
    'COSTS': {
        "doubao-seed-2-0-pro-260215": {"prompt": 0.0060/ 1000, "completion": 0.030 / 1000},
        "deepseek-v4-1-flash-economy-261017": {"prompt": 0.0016/ 1000, "completion": 0.0064 / 1000},
        "deepseek-v4-1-flash-261017": {"prompt": 0.002/ 1000, "completion": 0.008 / 1000},
        "doubao-seed-2-1-turbo-260628": {"prompt": 0.0030 / 1000, "completion": 0.015 / 1000},
        "doubao-seed-2-1-lite-260915": {"prompt": 0.0008 / 1000, "completion": 0.0027 / 1000},
        "doubao-embedding-vision-251215": {"prompt": 0.0007},
    }
}

# DOUBAO LLM配置（翻译与摘要）
DOUBAO_CONFIG = {
    'API_KEY': os.environ['ARK_API_KEY'],
    'BASE_URL': 'https://ark.cn-beijing.volces.com/api/v3',
    'MODEL_NAME': "doubao-seed-2-1-lite-260915",
    'REASONING_EFFORT': 'low',
}

# Embedding配置
EMBEDDING_CONFIG = {
    'API_KEY': os.environ['ARK_API_KEY'],
    'BASE_URL': 'https://ark.cn-beijing.volces.com/api/v3',
    'MODEL_NAME': "doubao-embedding-vision-251215", 
}

# 新闻服务API配置
NEWS_API_CONFIG = {
    'BASE_URL': 'http://10.3.8.16:9999/news_server/api',
    'RECALL_URL': 'http://10.3.8.16:9999/news_server/api/recallTask'
}

# 数据库配置
DB_CONFIG = {
    'host': os.environ['DB_HOST'],
    'user': os.environ['DB_USER'],
    'password': os.environ['DB_PASSWORD'],
    'database': os.environ['DB_NAME'],
}

# 向量搜索配置
VECTOR_SEARCH_CONFIG = {
    'DIMENSION': 4096, #1024,
    'THRESHOLD': 0.91,   # 相似度阈值越高，过滤掉的新闻越少，保留的新闻越多，建议为0.80-0.95
    'LOAD_HOURS': 48  # 导入最近48小时的数据
}
