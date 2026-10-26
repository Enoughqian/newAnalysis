"""Validate both HTTP transport status and the news server business acknowledgement."""
import time
import requests
from config import NEWS_API_CONFIG, RUNTIME_CONFIG


def fetch_tasks(taskname, count):
    # getTask may lease work, so do not retry a timed-out acquisition automatically.
    # Only an explicit 429/503 rejection is retried, with a short bounded delay.
    for attempt in range(3):
        response = requests.get(NEWS_API_CONFIG['BASE_URL'] + '/getTask',
                                params={'taskname': taskname, 'num': count, 'limit_time': 10},
                                timeout=(5, RUNTIME_CONFIG['HTTP_TIMEOUT']))
        if response.status_code in (429, 503) and attempt < 2:
            time.sleep(RUNTIME_CONFIG['RETRY_DELAY'] * (attempt + 1))
            continue
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError('任务接口未返回 JSON 对象')
        if body.get('err_code') not in (None, 0, '0'):
            raise ValueError(f'任务接口错误: {body.get("err_code")}')
        if body.get('msg') == '当前已取净':
            return []
        rows = body.get('data')
        if not isinstance(rows, list):
            raise ValueError('任务接口 data 必须是数组')
        if any(not isinstance(row, dict) or row.get('id') is None for row in rows):
            raise ValueError('任务缺少 id 或不是对象')
        if len({str(row['id']) for row in rows}) != len(rows):
            raise ValueError('任务列表含重复 id')
        return rows
    raise RuntimeError('获取任务失败')


def callback(taskname, rows):
    if not rows:
        return
    # Legacy backend rejects [] but accepts [''] and stores an empty country string.
    # Preserve semantic absence; do not invent a country label.
    wire_rows = [dict(row, result=['']) if taskname == 'recCountry' and row.get('result') == [] else row for row in rows]
    response = requests.post(NEWS_API_CONFIG['RECALL_URL'],
                             json={'taskname': taskname, 'data': wire_rows},
                             timeout=(5, RUNTIME_CONFIG['HTTP_TIMEOUT']))
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError('回调未返回 JSON 对象')
    if body.get('err_code') not in (None, 0, '0'):
        raise ValueError(f'{taskname} 回调业务错误: {body.get("err_code")}')
    count = body.get('success_num')
    if count is not None:
        if int(count) != len(rows):
            raise ValueError(f'{taskname} 回调只成功 {count}/{len(rows)} 条')
    elif body.get('err_code') not in (0, '0'):
        raise ValueError(f'{taskname} 回调缺少成功确认')

    return body
