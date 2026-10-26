import os

# LLM配置
LLM_CONFIG = {
    'API_KEY': '',
    "BASE_URL": "https://ark.cn-beijing.volces.com/api/v3",
    'MODEL_NAME': 'deepseek-v4-1-flash-economy-261017', #'doubao-seed-2-0-pro-260215',
    'REASONING_EFFORT': 'medium',
    'COSTS': {
        "doubao-seed-2-0-pro-260215": {"prompt": 0.0060/ 1000, "completion": 0.030 / 1000},
        "deepseek-v4-1-flash-economy-261017": {"prompt": 0.0016/ 1000, "completion": 0.0064 / 1000},
        "deepseek-v4-1-flash-261017": {"prompt": 0.002/ 1000, "completion": 0.008 / 1000},
        "doubao-seed-2-1-turbo-260628": {"prompt": 0.0030 / 1000, "completion": 0.015 / 1000},
        "doubao-seed-2-1-lite-260915": {"prompt": 0.0008 / 1000, "completion": 0.0027 / 1000},
        "doubao-embedding-vision-251215": {"prompt": 0.0007 / 1000},
    }
}

# DOUBAO LLM配置（翻译与摘要）
DOUBAO_CONFIG = {
    'API_KEY': '',
    'BASE_URL': 'https://ark.cn-beijing.volces.com/api/v3',
    'MODEL_NAME': "doubao-seed-2-1-lite-260915",
    'REASONING_EFFORT': 'low',
}

# Embedding配置
EMBEDDING_CONFIG = {
    'API_KEY': '',
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
    'host': '43.160.200.78',
    'user': 'work',
    'password': '',
    'database': 'spider_news'
}

# 向量搜索配置
VECTOR_SEARCH_CONFIG = {
    'DIMENSION': 2048,
    'THRESHOLD': 0.91,   # 相似度阈值越高，过滤掉的新闻越少，保留的新闻越多，建议为0.80-0.95
    'LOAD_HOURS': 48  # 导入最近48小时的数据
}

# 凭据只读取环境变量；run.sh 会加载本机 .env。
def env_bool(name, default=False):
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes"}

for prefix, settings in (("CLASSIFY", LLM_CONFIG), ("SUMMARY", DOUBAO_CONFIG),
                         ("EMBEDDING", EMBEDDING_CONFIG)):
    settings["API_KEY"] = os.getenv(prefix + "_API_KEY", os.getenv("ARK_API_KEY", settings["API_KEY"]))
    settings["BASE_URL"] = os.getenv(prefix + "_BASE_URL", os.getenv("ARK_BASE_URL", settings["BASE_URL"])).rstrip("/")
    settings["MODEL_NAME"] = os.getenv(prefix + "_MODEL", settings["MODEL_NAME"])
    if "REASONING_EFFORT" in settings:
        settings["REASONING_EFFORT"] = os.getenv(prefix + "_REASONING_EFFORT", settings["REASONING_EFFORT"])

# 空字符串表示省略 reasoning_effort，使用服务商默认行为。
EMBEDDING_CONFIG["API_TYPE"] = os.getenv("EMBEDDING_API_TYPE", "multimodal")
EMBEDDING_CONFIG["INSTRUCTIONS"] = os.getenv(
    "EMBEDDING_INSTRUCTIONS",
    "Target_modality: text.\nInstruction:Represent the news headline for semantic similarity.\nQuery:"
)
VECTOR_SEARCH_CONFIG["DIMENSION"] = int(os.getenv("EMBEDDING_DIMENSION", "2048"))
VECTOR_SEARCH_CONFIG["LOAD_HOURS"] = int(os.getenv("VECTOR_LOAD_HOURS", "48"))
VECTOR_SEARCH_CONFIG["THRESHOLD"] = float(os.getenv("VECTOR_THRESHOLD", "0.91"))
NEWS_API_CONFIG["BASE_URL"] = os.getenv("NEWS_API_BASE_URL", NEWS_API_CONFIG["BASE_URL"]).rstrip("/")
NEWS_API_CONFIG["RECALL_URL"] = NEWS_API_CONFIG["BASE_URL"] + "/recallTask"
for field in ("host", "user", "password", "database"):
    DB_CONFIG[field] = os.getenv("DB_" + field.upper(), DB_CONFIG[field])
DB_CONFIG["port"] = int(os.getenv("DB_PORT", "3306"))
DB_CONFIG["connection_timeout"] = 10
DB_CONFIG["read_timeout"] = 30
DB_CONFIG["write_timeout"] = 30

RUNTIME_CONFIG = {
    "STARTUP_DELAY_SECONDS": float(os.getenv("STARTUP_DELAY_SECONDS", "10")),
    "INTERVAL_SECONDS": float(os.getenv("INTERVAL_SECONDS", "600")),
    "SCHEDULER_ENABLED": env_bool("SCHEDULER_ENABLED", True),
    "DB_ENABLED": env_bool("DB_ENABLED", True),
    "HTTP_TIMEOUT": float(os.getenv("NEWS_HTTP_TIMEOUT", "15")),
    "MODEL_TIMEOUT": float(os.getenv("MODEL_TIMEOUT", "120")),
    "RETRY_DELAY": float(os.getenv("NEWS_RETRY_DELAY", "2")),
    "TITLE_BATCH_SIZE": int(os.getenv("TITLE_BATCH_SIZE", "20")),
    "CONTENT_BATCH_SIZE": int(os.getenv("CONTENT_BATCH_SIZE", "10")),
    "CONTENT_WORKERS": int(os.getenv("CONTENT_WORKERS", "5")),
    "TITLE_ROUNDS": int(os.getenv("TITLE_ROUNDS", "3")),
    "CONTENT_ROUNDS": int(os.getenv("CONTENT_ROUNDS", "2")),
}
