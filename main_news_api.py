"""News classification and summarization service; importing this module performs no I/O."""
import logging
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify

from config import LLM_CONFIG, DOUBAO_CONFIG, EMBEDDING_CONFIG, VECTOR_SEARCH_CONFIG, RUNTIME_CONFIG
from utils.pending_vectors import PendingVectors
from utils.database import VectorSearch
from utils.model_api import chat, embedding, estimate_cost, validate_model_config
from utils.news_api import fetch_tasks, callback
from utils.parse import parse_json, count_words, calculate_abstract_length
from utils.prompt_templates import classify_prompt, base_content_prompt, short_content_prompt, country_normalization_map

logger = logging.getLogger(__name__)
app = Flask(__name__)
scheduler = BackgroundScheduler(daemon=True, timezone='Asia/Shanghai')
# All manual triggers, scheduled runs and index refreshes share one process-wide lock.
processing_lock = threading.Lock()
ready = False
last_pipeline = None
faiss_matrix = VectorSearch(VECTOR_SEARCH_CONFIG['DIMENSION'], logger)
pending_vectors = PendingVectors()


def validate_classification(text, count):
    ok, result = parse_json(text)
    if not ok or not isinstance(result, dict):
        raise ValueError('分类结果不是合法 JSON 对象')
    result = {str(k): v for k, v in result.items()}
    if set(result) != {str(i) for i in range(1, count + 1)}:
        raise ValueError('分类结果序号缺失或越界')
    for item in result.values():
        if not isinstance(item, dict) or type(item.get('flag')) is not bool:
            raise ValueError('分类 flag 必须是 JSON 布尔值')
        if item['flag'] and item.get('theme') not in {'政治', '军事', '社会', '经济'}:
            raise ValueError('分类 theme 必须是四种主题之一')
    return result


def validate_summary(text, max_abstract_len=300):
    ok, result = parse_json(text)
    if not ok or not isinstance(result, dict):
        raise ValueError('摘要结果不是合法 JSON 对象')
    for field in ('title', 'abstract'):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ValueError('摘要缺少 ' + field)
    for field in ('keywords', 'country'):
        if not isinstance(result.get(field), list) or any(not isinstance(x, str) for x in result[field]):
            raise ValueError(field + ' 必须是字符串数组')
    return result


class NewsAnalyzer:
    def analyze_news_list(self):
        global faiss_matrix
        cost = 0.0
        try:
            rows = fetch_tasks('recTitle', RUNTIME_CONFIG['TITLE_BATCH_SIZE'])
            if not rows:
                return {'status': 'success', 'message': '无待分类新闻', 'processed_count': 0, 'cost': 0.0}
            if any(not isinstance(row.get('title'), str) or not row['title'].strip() for row in rows):
                raise ValueError('待分类新闻标题为空')
            import json
            text, usage = chat(classify_prompt + json.dumps([r['title'] for r in rows], ensure_ascii=False),
                               '你是新闻分类与判断专家', LLM_CONFIG)
            cost = estimate_cost(LLM_CONFIG['MODEL_NAME'], usage)
            classifications = validate_classification(text, len(rows))
            staged = faiss_matrix.clone()
            results, vectors = [], []
            classify_cost = cost / len(rows)
            for i, row in enumerate(rows, 1):
                answer = classifications[str(i)]
                vector_cost = 0.0
                tag = 0
                if answer['flag']:
                    vector, vector_usage = embedding(row['title'])
                    vector_cost = estimate_cost(EMBEDDING_CONFIG['MODEL_NAME'], vector_usage)
                    cost += vector_cost
                    tag = 1 if staged.add_vector(vector, news_id=row['id']) else -1
                    if tag == 1:
                        vectors.append({'id': row['id'], 'result': vector, 'cost': round(vector_cost, 8)})
                item = {'id': row['id'], 'tag': tag, 'cost': round(classify_cost + vector_cost, 8)}
                if tag:
                    item['classify'] = [answer['theme']]
                results.append(item)
            # The online crawler creates news_detail only after recTitle is accepted.
            # Persist a model-scoped hand-off locally; do not send genVec to a missing row.
            pending_vectors.save(vectors)
            callback('recTitle', results)
            pending_vectors.confirm([row['id'] for row in vectors])
            faiss_matrix = staged
            return {'status': 'success', 'message': '新闻分类已完成', 'processed_count': len(results),
                    'cost': round(cost, 8), 'detailed_results': results}
        except Exception as exc:
            logger.exception('新闻分类失败')
            return {'status': 'error', 'message': str(exc), 'processed_count': 0, 'cost': round(cost, 8)}

    def _get_content_list(self):
        return fetch_tasks('recCountry', RUNTIME_CONFIG['CONTENT_BATCH_SIZE'])

    def process_single_content(self, row):
        result = {'id': row['id'], 'status': 'success', 'cost': 0.0,
                  'prompt_tokens': 0, 'completion_tokens': 0, 'abstract_len': 0,
                  'original_len': len(row.get('content') or ''), 'errors': [], 'failed_callbacks': []}
        try:
            content = row.get('content')
            if not isinstance(content, str) or not content.strip():
                raise ValueError('empty_content')
            length = count_words(content)
            if length < 250:
                prompt = short_content_prompt
                maximum = 300
            else:
                minimum = calculate_abstract_length(length)
                maximum = min(minimum * 2, 300)
                prompt = base_content_prompt.format(min_abstract_len=minimum, max_abstract_len=maximum)
            text, usage = chat(prompt + content, '你是新闻阅读与结构化整理专家', DOUBAO_CONFIG)
            cost = estimate_cost(DOUBAO_CONFIG['MODEL_NAME'], usage)
            total_prompt = usage.get('prompt_tokens', 0)
            total_completion = usage.get('completion_tokens', 0)
            answer = validate_summary(text)
            result.update(cost=round(cost, 8), prompt_tokens=total_prompt,
                          completion_tokens=total_completion)
            abstract = answer['title'] + '\n' + answer['abstract']
            result['abstract_len'] = len(abstract)
            countries = []
            for country in answer['country']:
                country = country.strip().split('(')[0].split('（')[0]
                country = country_normalization_map.get(country, country)
                if country and country not in countries:
                    countries.append(country)
            # recCountry is the acquisition queue: only acknowledge it after all earlier writes succeed.
            tasks = [('genAbstract', abstract), ('genKeyword', answer['keywords']), ('recCountry', countries)]
            cached_vector = pending_vectors.get(row['id'])
            if cached_vector is not None:
                tasks.insert(0, ('genVec', cached_vector))
            for task, value in tasks:
                try:
                    callback(task, [{'id': row['id'], 'result': value, 'cost': 0.0 if task == 'genVec' else round(cost / 3, 8)}])
                    if task == 'genVec':
                        pending_vectors.delete(row['id'])
                except Exception:
                    result['failed_callbacks'].append(task)
                    result['errors'].append('callback_' + task + '_failed')
                    result['status'] = 'partial_success'
                    logger.exception('新闻 %s 的 %s 回写失败', row['id'], task)
                    break
        except Exception as exc:
            result['status'] = 'failed'
            result['errors'].append(str(exc))
            logger.exception('内容处理失败: %s', row['id'])
        return result

    def batch_process_contents(self, batch_size=None):
        started = time.monotonic()
        try:
            rows = self._get_content_list()
            if not rows:
                return {'status': 'success', 'message': '无待处理正文', 'processed_count': 0,
                        'statistics': {'total': 0, 'success': 0, 'partial_success': 0, 'failed': 0, 'total_cost': 0},
                        'detailed_results': []}
            with ThreadPoolExecutor(max_workers=batch_size or RUNTIME_CONFIG['CONTENT_WORKERS']) as executor:
                futures = [executor.submit(self.process_single_content, row) for row in rows]
                results = [future.result() for future in as_completed(futures)]
            statuses = Counter(r['status'] for r in results)
            errors = Counter(e for r in results for e in r['errors'])
            callback_errors = Counter(t for r in results for t in r['failed_callbacks'])
            status = 'success' if statuses['success'] == len(results) else ('error' if statuses['failed'] == len(results) else 'partial_success')
            return {'status': status, 'processed_count': len(results), 'statistics': {
                'total': len(results), 'success': statuses['success'], 'partial_success': statuses['partial_success'],
                'failed': statuses['failed'], 'total_cost': sum(r['cost'] for r in results),
                'avg_processing_time': (time.monotonic() - started) / len(results),
                'common_errors': dict(errors),
                'callback_failure_rates': {k: f'{v / len(results):.1%}' for k, v in callback_errors.items()}},
                'detailed_results': results}
        except Exception as exc:
            logger.exception('内容批次失败')
            return {'status': 'error', 'message': str(exc), 'processed_count': 0}


news_analyzer = NewsAnalyzer()


def run_pipeline():
    global last_pipeline
    if not ready or not processing_lock.acquire(blocking=False):
        logger.info('跳过本轮：尚未就绪或已有任务执行')
        return
    try:
        reports = []
        for _ in range(RUNTIME_CONFIG['TITLE_ROUNDS']):
            report = news_analyzer.analyze_news_list()
            reports.append(report)
            if report['status'] != 'success' or not report.get('processed_count'):
                break
        for _ in range(RUNTIME_CONFIG['CONTENT_ROUNDS']):
            report = news_analyzer.batch_process_contents()
            reports.append(report)
            if report['status'] != 'success' or not report.get('processed_count'):
                break
        last_pipeline = {'finished_at': datetime.now(timezone.utc).isoformat(),
                         'status': 'success' if all(r['status'] == 'success' for r in reports) else 'error',
                         'reports': reports}
        logger.info('本轮分类与摘要完成: %s', last_pipeline['status'])
    finally:
        processing_lock.release()


def refresh_faiss_matrix():
    global faiss_matrix
    if not RUNTIME_CONFIG['DB_ENABLED'] or not processing_lock.acquire(blocking=False):
        return
    try:
        from add_embedding_time import add_missing_embeddings
        result = add_missing_embeddings(hours=VECTOR_SEARCH_CONFIG['LOAD_HOURS'])
        if result['fail']:
            raise RuntimeError('部分历史向量补充失败，保留当前索引')
        staged = VectorSearch(VECTOR_SEARCH_CONFIG['DIMENSION'], logger)
        staged.import_from_mysql(VECTOR_SEARCH_CONFIG['LOAD_HOURS'])
        for ident, vector in pending_vectors.recent(VECTOR_SEARCH_CONFIG['LOAD_HOURS']):
            staged.add_vector(vector, news_id=ident)
        faiss_matrix = staged
    except Exception:
        logger.exception('向量刷新失败，保留当前索引')
    finally:
        processing_lock.release()


def initialize_service():
    global ready, faiss_matrix
    if ready:
        return
    validate_model_config()
    for key in ('INTERVAL_SECONDS', 'HTTP_TIMEOUT', 'MODEL_TIMEOUT', 'TITLE_BATCH_SIZE', 'CONTENT_BATCH_SIZE',
                'CONTENT_WORKERS', 'TITLE_ROUNDS', 'CONTENT_ROUNDS'):
        if RUNTIME_CONFIG[key] <= 0:
            raise ValueError(key + ' 必须大于零')
    if RUNTIME_CONFIG['STARTUP_DELAY_SECONDS'] < 0:
        raise ValueError('启动延迟不可为负数')
    if RUNTIME_CONFIG['DB_ENABLED']:
        staged = VectorSearch(VECTOR_SEARCH_CONFIG['DIMENSION'], logger)
        staged.import_from_mysql(VECTOR_SEARCH_CONFIG['LOAD_HOURS'])
        faiss_matrix = staged
    for ident, vector in pending_vectors.recent(VECTOR_SEARCH_CONFIG['LOAD_HOURS']):
        faiss_matrix.add_vector(vector, news_id=ident)
    ready = True
    logger.info('服务初始化完成，历史向量导入=%s，索引条数=%s',
                RUNTIME_CONFIG['DB_ENABLED'], faiss_matrix.display_embedding_nums())


def start_scheduler():
    if not RUNTIME_CONFIG['SCHEDULER_ENABLED'] or scheduler.running:
        return
    if not ready:
        raise RuntimeError('请先初始化服务')
    scheduler.add_job(run_pipeline, 'interval', seconds=RUNTIME_CONFIG['INTERVAL_SECONDS'],
                      next_run_time=datetime.now(timezone.utc) + timedelta(seconds=RUNTIME_CONFIG['STARTUP_DELAY_SECONDS']),
                      id='news_pipeline', max_instances=1, coalesce=True, misfire_grace_time=60)
    if RUNTIME_CONFIG['DB_ENABLED']:
        scheduler.add_job(refresh_faiss_matrix, 'interval', hours=1, id='refresh_vectors',
                          max_instances=1, coalesce=True)
    scheduler.start()
    logger.info('首次任务等待 %s 秒，此后每 %s 秒执行',
                RUNTIME_CONFIG['STARTUP_DELAY_SECONDS'], RUNTIME_CONFIG['INTERVAL_SECONDS'])


def configure_logging():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s-%(asctime)s] %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            os.getenv('NEWS_LOG_FILE', 'news_analysis.log'), encoding='utf-8')])


@app.get('/healthz')
def health():
    job = scheduler.get_job('news_pipeline')
    return jsonify({'status': 'ready' if ready else 'not_ready', 'scheduler_running': scheduler.running,
                    'startup_delay_seconds': RUNTIME_CONFIG['STARTUP_DELAY_SECONDS'],
                    'interval_seconds': RUNTIME_CONFIG['INTERVAL_SECONDS'],
                    'next_run_time': job.next_run_time.isoformat() if job and scheduler.running else None,
                    'history_loaded': RUNTIME_CONFIG['DB_ENABLED'] and ready,
                    'vector_count': faiss_matrix.display_embedding_nums(),
                    'models': {'classify': LLM_CONFIG['MODEL_NAME'], 'summary': DOUBAO_CONFIG['MODEL_NAME'],
                               'embedding': EMBEDDING_CONFIG['MODEL_NAME']}, 'last_pipeline': last_pipeline}), 200 if ready else 503


def trigger(method):
    if not ready:
        return jsonify({'status': 'error', 'message': '服务尚未初始化'}), 503
    if not processing_lock.acquire(blocking=False):
        return jsonify({'status': 'busy', 'message': '已有任务执行，请稍后重试'}), 409
    try:
        result = method()
        return jsonify(result), 200 if result['status'] == 'success' else 502
    finally:
        processing_lock.release()


# Retain existing GET clients; POST is preferred for operations that change task state.
@app.route('/trigger/news', methods=['GET', 'POST'])
def trigger_news_analysis():
    return trigger(news_analyzer.analyze_news_list)


@app.route('/trigger/content', methods=['GET', 'POST'])
def trigger_content_processing():
    return trigger(news_analyzer.batch_process_contents)


if __name__ == '__main__':
    configure_logging()
    initialize_service()
    start_scheduler()
    try:
        app.run(host=os.getenv('HOST', '127.0.0.1'), port=int(os.getenv('PORT', '2400')), use_reloader=False)
    finally:
        if scheduler.running:
            scheduler.shutdown(wait=False)
