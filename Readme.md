# 新闻分类、去重与摘要服务

Flask + Gunicorn + APScheduler。默认在服务初始化、历史向量加载完成后 **等待 10 秒**开始第一轮，之后 **每 10 分钟**自动执行“标题分类 → 正文摘要、关键词、国家识别”。

字数只作提示和统计，不因超长拦截、不机械截断、不自动压缩重试。正文结果通过线上新闻服务 `recallTask` 写回。

## 1. 凭据放在哪里

- `config.py` 保留模型、接口、数据库和运行参数，API Key、数据库密码只读环境变量。
- **`.env`** 保存本机真实值，已被 Git 忽略，不提交。
- **`.env.example`** 是可提交的空值模板。
- `run.sh` 自动加载 `.env`；直接执行 Python/Gunicorn 时，需要先导出环境变量。

```bash
cp .env.example .env
chmod 600 .env
# 编辑 .env，填入：
# ARK_API_KEY='你的方舟 API Key'
# DB_PASSWORD='你的数据库密码'
```

可选独立 Key：`CLASSIFY_API_KEY`、`SUMMARY_API_KEY`、`EMBEDDING_API_KEY`。不要把真实凭据写回 `config.py`。旧版本已提交过的凭据不会因新版本移到 `.env` 而从 Git 历史消失，建议另行轮换。

## 2. 服务器更新到本分支

建议 Python 3.13，Linux 需安装 Python venv 支持及 FAISS 使用的 OpenMP 运行库（Debian/Ubuntu 通常是 `python3-venv`、`libgomp1`）。不要把 Mac 的 `.venv` 复制到 Linux；在服务器重新创建环境。

先用现有的进程管理方式停止旧服务，避免同时运行新旧实例。在服务器的 Git 项目目录操作：

```bash
# 保存旧配置，后续从中提取凭据；此文件已加入 Git 忽略规则
cp -p config.py config.py.legacy
chmod 600 config.py.legacy

git fetch https://github.com/suay1113/newAnalysis.git fix/model-adaptation-scheduler
# 如果 config.py 有本地修改：确认上面的备份存在后，仅恢复该文件
# git restore -- config.py

# 首次切换到 Fork 中的 PR 分支
git switch -c fix/model-adaptation-scheduler FETCH_HEAD
# 若本地已有此分支，改为：
# git switch fix/model-adaptation-scheduler
# git merge --ff-only FETCH_HEAD

# 从旧文件提取凭据、数据库地址和新闻服务地址到 .env，不打印值
python3 scripts/migrate_env.py --source config.py.legacy --output .env
# 已有 .env 的非空值不会被覆盖；可自行编辑 .env

# 安装已测试版本依赖，然后运行完全本地的测试
./run.sh setup
./run.sh test
```

PR 合并后，正常更新可使用 `git switch main && git pull --ff-only origin main`。如果 Git 提示其他源文件有本地修改，保留这些改动，先合并处理，不要用强制重置覆盖。

确认 `.env` 中以下设置：

```bash
SCHEDULER_ENABLED=1
STARTUP_DELAY_SECONDS=10
INTERVAL_SECONDS=600
DB_ENABLED=1
MODEL_TIMEOUT=240
BIND=127.0.0.1:2400
```

服务器默认新闻服务地址仍是 `config.py` 中的内网地址；需要更换时设置 `NEWS_API_BASE_URL`。数据库可以分别用 `DB_HOST`、`DB_USER`、`DB_PASSWORD`、`DB_DATABASE`、`DB_PORT` 覆盖。

## 3. 启动、查看和停止

前台启动：

```bash
./run.sh serve
```

普通 Linux 服务器后台运行：

```bash
mkdir -p .cache
nohup ./run.sh serve > news-analysis.out.log 2>&1 &
echo $! > .cache/news-analysis.pid

# 查看日志和状态
tail -f news-analysis.out.log
curl -s http://127.0.0.1:2400/healthz

# 停止由上述命令启动的实例
kill -TERM "$(cat .cache/news-analysis.pid)"
```

已有 systemd/Supervisor 的服务器可继续使用原进程管理器：将工作目录设置为项目目录，启动命令设为该目录下的 `run.sh serve`，以同一个有权读取 `.env` 和写入 `.cache` 的用户运行。只保留一个实例。`nohup` 不提供开机自启，开机自启由原进程管理器配置。

`healthz` 应显示：

```json
{
  "status": "ready",
  "scheduler_running": true,
  "startup_delay_seconds": 10.0,
  "interval_seconds": 600.0,
  "next_run_time": "下一轮计划时间"
}
```

首次 10 秒从初始化完成、调度器启动时开始计算，不包含依赖安装和数据库加载时间。每轮最多分类 3×20 条，再处理正文 2×10 条。后续间隔按计划触发时间计算；上一轮仍在执行时不会再并发叠加一轮，过期轮次合并处理。

**必须使用提供的 `gunicorn.conf.py`**（`run.sh serve` 已自动使用）。它以一个 worker、四个线程运行，并在 worker 初始化后启动调度器。不要沿用旧的 `gunicorn -w 4 main_news_api:app`；不要使用 `--preload`。开发调试可用 `./run.sh dev`。

## 4. 模型与向量

| 用途 | 默认模型 | 参数 |
|---|---|---|
| 标题分类 | `deepseek-v4-1-flash-economy-261017` | `reasoning_effort=medium` |
| 摘要、关键词、国家 | `doubao-seed-2-1-lite-260915` | `reasoning_effort=low` |
| 标题向量 | `doubao-embedding-vision-251215` | 多模态接口中的纯文本输入，2048 维 |

配置对应 `CLASSIFY_MODEL`、`SUMMARY_MODEL`、`EMBEDDING_MODEL`；独立地址用 `CLASSIFY_BASE_URL`、`SUMMARY_BASE_URL`、`EMBEDDING_BASE_URL`。修改后重启服务。更新模型时也要检查 `LLM_CONFIG['COSTS']` 中的每 token 单价；费用为配置单价估算，不代替服务商账单。

向量使用 `/embeddings/multimodal` 的 `data.embedding`；要求数值有效、维度匹配。数据库保持 float64 存储，2048 维对应 16384 字节；FAISS 使用归一化 float32。最近 48 小时的历史向量在启动时加载，每小时补充缺失向量并刷新。

更换向量模型、维度或 instructions 时，应统一重建参与去重的历史范围，不能截断/补零，也不能混用同维度但不同模型的向量。程序会检查实际数据，不再要求额外填写历史迁移确认开关。

```bash
# 先加载 .env；下面命令默认只统计，不回写
set -a
source .env
set +a
.venv/bin/python add_embedding_time.py --rebuild --hours 48

# 只有需要重建时才执行；先保存旧向量，备份文件必须不存在
.venv/bin/python add_embedding_time.py --rebuild --hours 48 --apply \
  --backup embedding-backup-before-migration.jsonl
```

本次线上积压测试中的 76 条向量已更新为 2048 维并核验，部署时不必盲目重写全库；如果启动检查发现其他不匹配数据，再核对具体范围。

标题确认后线上爬虫才创建正文。因此分类向量先保存到 `.cache/pending-vectors.sqlite3`，正文阶段再回写 `genVec`。该交接表按模型、维度、接口和 instructions 隔离，进程重启可恢复。迁移机器时，保留待交接的 `.cache` 数据，或等正文创建后重新补向量。

## 5. 手动触发与错误处理

```bash
curl -X POST http://127.0.0.1:2400/trigger/news
curl -X POST http://127.0.0.1:2400/trigger/content
# 旧 GET 客户端仍兼容；测试脚本默认只调用一次
.venv/bin/python test-title.py
.venv/bin/python test-content.py
```

接口同步完成后返回。HTTP 200 表示本批成功或无任务；502 表示模型/解析/回写失败或部分成功；409 表示已有任务执行；503 表示尚未初始化。空队列立即返回，不再等待 180 秒。

回调同时检查 HTTP 与业务确认。正文依次写 `genVec`（如有待交接向量）、`genAbstract`、`genKeyword`、`recCountry`；前一步失败不确认后续队列完成。国家为空时使用旧后端接受的空值编码，线上保存空字符串，不填造国家。标题队列的空 `msg` + 空 `data` 正常视为无任务。

模型超时由 `MODEL_TIMEOUT` 控制；新闻接口超时由 `NEWS_HTTP_TIMEOUT` 控制。回调之间不是跨服务事务；重试和任务恢复依赖后端按 ID 幂等回写。费用统计不会重复计算缓存 token。`limit_time=10` 是原有任务锁参数，不是 HTTP 超时。

## 6. 测试

```bash
# 无外网模型费用、无业务数据修改：回归 + Python HTTP E2E + Gunicorn 定时 E2E
./run.sh test

# 三个真实模型各一次小请求，会产生少量费用，不写业务数据
./run.sh models

# 真实模型 + 本地合成新闻和回写后端
./run.sh e2e --live-models --output test-results/e2e-live-models

# 只读检查业务服务连通性、数据库向量长度分布
./run.sh doctor
```

自动测试覆盖分类、去重、长短正文、费用、回调失败、空队列、空国家、持久化向量交接、并发锁，以及首次 10 秒执行和后续 600 秒计划。测试默认不持续等待第二个 10 分钟周期。

2026-10-26 另完成了 76 条真实积压新闻的线上回写和逐条数据库核验，最终全部通过；没有重写已完成的历史新闻。详细原文、模型输出、凭据文件、备份和本地审计报告不随代码上传。测试结果写入被 Git 忽略的 `test-results/`。
