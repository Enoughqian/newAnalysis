"""One worker owns the scheduler and FAISS index; threads allow health checks during work."""
import os
workers = 1
worker_class = 'gthread'
threads = 4
bind = os.getenv('BIND', '127.0.0.1:2400')
timeout = 600
preload_app = False


def on_starting(server):
    if server.cfg.workers != 1 or server.cfg.preload_app:
        raise RuntimeError('当前实现需要 workers=1 且 preload_app=False，避免重复领取任务与独立向量索引')


def post_worker_init(worker):
    from main_news_api import configure_logging, initialize_service, start_scheduler
    configure_logging()
    initialize_service()
    start_scheduler()


def worker_exit(server, worker):
    from main_news_api import scheduler
    if scheduler.running:
        scheduler.shutdown(wait=False)
