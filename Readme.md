# 新闻摘要与分类系统

基于火山引擎豆包大模型的新闻自动分类与摘要系统。

## 模型配置

| 功能 | 模型 | reasoning_effort | 说明 |
|------|------|-----------------|------|
| 新闻标题分类 | doubao-seed-2-0-pro-260215 | medium | 判断新闻是否属于关注领域并归类主题 |
| 摘要与提取 | doubao-seed-2-0-lite-260215 | low | 摘要生成、关键词提取、国家识别 |
| Embedding | doubao-embedding-large-text-240915 | - | 标题向量化，用于 faiss 去重 |

所有模型均通过火山引擎方舟平台（`ark.cn-beijing.volces.com`）调用，使用 OpenAI 兼容接口。

## 环境安装

```bash
pip install -r requirements.txt
```

## 启动服务

```bash
# 生产环境
gunicorn -w 4 -b 0.0.0.0:2400 main_news_api:app

# 开发调试
python main_news_api.py
```

服务默认端口 2400，启动后自动运行定时任务（每10分钟执行分类和摘要）。

## 接口与测试

### 新闻分类

```bash
# 单次调用
curl http://localhost:2400/trigger/news

# 批量测试（循环60次）
python test-title.py
```

### 摘要与提取

```bash
# 单次调用
curl http://localhost:2400/trigger/content

# 批量测试（循环30次）
python test-content.py
```

## 摘要输出格式

摘要按以下结构生成：

```
标题（不超过20字，句末加句号）
据【信息来源】X月X日报道，……
第二段内容。
```

- 标题：简洁中文标题，体现核心新闻价值，不直译原标题
- 正文首句：标注信息来源与日期
- 总字数约200字，分为2-4个自然段，每段不超过50字
- 专有名词首次出现时标注英文全称

## 项目结构

```
├── main_news_api.py          # 主服务，Flask API + 定时任务
├── config.py                 # 模型、数据库、API 等配置
├── add_embedding_time.py     # 补充缺失 embedding 的脚本
├── utils/
│   ├── prompt_templates.py   # Prompt 模板与国家名称映射
│   ├── parse.py              # JSON 解析、字数统计等工具函数
│   └── database.py           # faiss 向量搜索与 MySQL 导入
├── test-title.py             # 分类接口批量测试
├── test-content.py           # 摘要接口批量测试
└── requirements.txt
```

## 设计说明

1. 所有大模型服务基于火山引擎方舟平台，使用 Seed 2.0 系列推理模型，通过 `reasoning_effort` 参数控制推理深度以平衡质量与成本
2. Embedding 在标题分类阶段调用，使用 faiss 进行去重，相似度阈值 0.91
3. 分类使用 Seed 2.0 Pro（medium 推理），摘要使用 Seed 2.0 Lite（low 推理）以降低成本
4. 已移除全文翻译功能，仅保留摘要生成，大幅降低 completion tokens 消耗
5. news 接口一次处理 20 条标题分类，content 接口一次处理 10 条摘要与提取
6. Embedding 仅导入最近 48 小时数据，每 1 小时自动刷新并补充缺失向量
7. 手动补充 embedding：`python add_embedding_time.py`