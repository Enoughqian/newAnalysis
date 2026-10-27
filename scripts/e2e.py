#!/usr/bin/env python3
"""Real HTTP E2E. Default: local model + news fixtures. --live-models: real Ark, local news only."""
import argparse
import copy
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import requests
from flask import Flask, jsonify, request
from werkzeug.serving import make_server
from config import LLM_CONFIG, DOUBAO_CONFIG, EMBEDDING_CONFIG, VECTOR_SEARCH_CONFIG

ROWS = [
    {'id':'e2e-1', 'title':'Saudi Arabia announces a new national defence procurement programme',
     'content':'Saudi Arabia announced a national defence procurement programme on Monday. The defence ministry said the programme would modernise military equipment and improve readiness. Further contract details were not disclosed.'},
    {'id':'e2e-2', 'title':'Saudi Arabia announces a new national defence procurement programme',
     'content':'Duplicate test article.'},
    {'id':'e2e-3', 'title':'Tokyo hosts a local cooking competition', 'content':'A local cooking contest took place in Tokyo.'},
    {'id':'e2e-4', 'title':'Pakistan and Saudi Arabia sign an agreement to expand bilateral trade',
     'content':('Pakistan and Saudi Arabia signed an agreement on Monday to expand bilateral trade. Officials said the agreement would simplify customs procedures and improve cooperation between exporters. The two governments will establish a joint committee to monitor implementation. No financial value or binding investment total was disclosed. ' * 6)},
]


class Fixture:
    def __init__(self, output):
        self.output = output
        self.lock = threading.Lock()
        self.reset()
        self.app = Flask('news_e2e_fixture')
        self.app.add_url_rule('/news_server/api/getTask', view_func=self.get_tasks)
        self.app.add_url_rule('/news_server/api/recallTask', view_func=self.recall, methods=['POST'])
        self.app.add_url_rule('/api/v3/chat/completions', view_func=self.chat, methods=['POST'])
        self.app.add_url_rule('/api/v3/embeddings/multimodal', view_func=self.embed, methods=['POST'])

    def reset(self):
        self.records = {r['id']: copy.deepcopy(r) for r in ROWS}
        self.callbacks = []
        self.model_calls = []
        self.fail_task = None
        self.fail_kind = 'http'
        self.invalid_json = False
        self.bad_dimension = False
        self.first_task_at = None

    def get_tasks(self):
        if self.first_task_at is None:
            self.first_task_at = time.monotonic()
        task = request.args['taskname']
        if task == 'recTitle':
            rows = [r for r in self.records.values() if 'tag' not in r]
        else:
            rows = [r for r in self.records.values() if r.get('tag') == 1 and 'recCountry' not in r]
        rows = rows[:int(request.args['num'])]
        return jsonify({'msg':'获取完成' if rows else '当前已取净','data':rows,'num':len(rows),'err_code':0})

    def recall(self):
        with self.lock:
            return self._recall()

    def _recall(self):
        body = request.get_json()
        task = body['taskname']
        if task == self.fail_task:
            if self.fail_kind == 'http':
                return jsonify({'err_code':500}), 500
            return jsonify({'err_code':1,'success_num':0})
        for row in body['data']:
            record = self.records[row['id']]
            if task == 'recTitle':
                record.update(row)
            else:
                record[task] = row['result']
        self.callbacks.append(task)
        # Persist fixture state so E2E validates stored results, not only API success.
        self.output.write_text(json.dumps(self.records,ensure_ascii=False,indent=2))
        return jsonify({'err_code':0,'success_num':len(body['data'])})

    def chat(self):
        body = request.get_json()
        self.model_calls.append({'path':'chat','model':body['model'], 'reasoning_effort':body.get('reasoning_effort')})
        classify = '分类' in body['messages'][0]['content']
        if classify:
            titles = json.loads(body['messages'][1]['content'].split('### 待检测的新闻标题',1)[1].strip())
            result = {str(i): ({'flag':False} if 'Tokyo' in title else
                      {'flag':True,'theme':'经济' if 'trade' in title else '军事'}) for i,title in enumerate(titles,1)}
        else:
            result = {'title':'两国加强合作。','abstract':'有关部门宣布开展合作，具体安排将由双方进一步协调。',
                      'keywords':['合作','协议','政策'],'country':['沙特阿拉伯']}
        content = 'not-json' if self.invalid_json else json.dumps(result,ensure_ascii=False)
        return jsonify({'id':'fixture','object':'chat.completion','created':0,'model':body['model'],
                        'choices':[{'index':0,'message':{'role':'assistant','content':content},'finish_reason':'stop'}],
                        'usage':{'prompt_tokens':100,'completion_tokens':30,'total_tokens':130}})

    def embed(self):
        body=request.get_json()
        self.model_calls.append({'path':'embedding','model':body['model']})
        assert body['input'][0]['type']=='text'
        assert body['dimensions']==VECTOR_SEARCH_CONFIG['DIMENSION']
        vector=[0.0] * (3 if self.bad_dimension else body['dimensions'])
        vector[1 if 'trade' in body['input'][0]['text'] else 0]=1.0
        return jsonify({'data':{'embedding':vector},'usage':{'prompt_tokens':20,'total_tokens':20}})


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run(args):
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    fixture=Fixture(output/'stored-news.json')
    fixture_server=make_server('127.0.0.1',0,fixture.app,threaded=True)
    thread=threading.Thread(target=fixture_server.serve_forever,daemon=True)
    thread.start()
    fixture_base=f'http://127.0.0.1:{fixture_server.server_port}'
    service_port=port()
    base=f'http://127.0.0.1:{service_port}'
    (output/'models.jsonl').write_text('')
    env=os.environ.copy()
    env.update(DB_ENABLED='0',SCHEDULER_ENABLED='1' if args.scheduled else '0',
               STARTUP_DELAY_SECONDS='10', INTERVAL_SECONDS='600',
               NEWS_API_BASE_URL=fixture_base+'/news_server/api', HOST='127.0.0.1',PORT=str(service_port),
               BIND=f'127.0.0.1:{service_port}', TITLE_BATCH_SIZE='20',CONTENT_BATCH_SIZE='10',
               TITLE_ROUNDS='3',CONTENT_ROUNDS='2',NEWS_RETRY_DELAY='0',
               MODEL_AUDIT_FILE=str(output/'models.jsonl'), PENDING_VECTOR_DB=str(output/'pending.sqlite3'),NEWS_LOG_FILE=str(output/'service.log'),NO_PROXY='127.0.0.1,localhost')
    if not args.live_models:
        for prefix in ('CLASSIFY','SUMMARY','EMBEDDING'):
            env[prefix+'_BASE_URL']=fixture_base+'/api/v3'
            env[prefix+'_API_KEY']='offline-test-key'
    command=([sys.executable,'-m','gunicorn','-c','gunicorn.conf.py','main_news_api:app']
             if args.server=='gunicorn' else [sys.executable,'main_news_api.py'])
    session=requests.Session(); session.trust_env=False
    process=None
    started=time.monotonic()
    try:
        with (output/'process.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=log)
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            if process.poll() is not None:
                raise RuntimeError('服务启动失败；见 '+str(output/'process.log'))
            try:
                health=session.get(base+'/healthz',timeout=2)
                if health.status_code==200:
                    break
            except requests.RequestException:
                pass
            time.sleep(.1)
        else:
            raise RuntimeError('服务启动超时')
        h=health.json()
        require(h['startup_delay_seconds']==10 and h['interval_seconds']==600,'调度配置错误')
        checks=['service_started','health_ready']
        if args.scheduled:
            require(h['scheduler_running'],'调度器没有启动')
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                h=session.get(base+'/healthz',timeout=3).json()
                if h['last_pipeline']:
                    break
                time.sleep(.2)
            else:
                raise AssertionError('首次自动任务未完成')
            require(fixture.first_task_at-started>=9.5,'首次任务过早执行')
            require(h['last_pipeline']['status']=='success','自动任务失败')
            from datetime import datetime
            remaining=(datetime.fromisoformat(h['next_run_time'])-datetime.now().astimezone()).total_seconds()
            require(0<remaining<=600,'下一轮执行时间错误')
            (output/'schedule-health.json').write_text(json.dumps(h,ensure_ascii=False,indent=2))
            checks += ['first_run_after_10_seconds','next_run_interval_600_seconds']
        else:
            require(not h['scheduler_running'],'手动测试意外开启调度器')
            title=session.post(base+'/trigger/news',timeout=600)
            (output/'classification-response.json').write_text(json.dumps(title.json(),ensure_ascii=False,indent=2))
            require(title.status_code==200, '分类失败: '+title.text)
            require(title.json()['processed_count']==4,'分类数量错误')
            require(title.json()['cost']>0,'费用未记录')
            content=session.post(base+'/trigger/content',timeout=600)
            (output/'content-response.json').write_text(json.dumps(content.json(),ensure_ascii=False,indent=2))
            require(content.status_code==200,'摘要失败: '+content.text)
            require(content.json()['statistics']['success']==2,'摘要成功数量错误')
        stored=json.loads((output/'stored-news.json').read_text())
        (output/'successful-news.json').write_text(json.dumps(stored,ensure_ascii=False,indent=2))
        require([stored[f'e2e-{i}']['tag'] for i in range(1,5)]==[1,-1,0,1],'分类或去重结果不符')
        for key in ('e2e-1','e2e-4'):
            for field in ('genAbstract','genKeyword','recCountry'):
                require(bool(stored[key].get(field)),f'{key} 未保存 {field}')
            require(len(stored[key]['genVec'])==VECTOR_SEARCH_CONFIG['DIMENSION'],'存储向量维度错误')
        checks += ['classification','duplicate_filtered','irrelevant_filtered','short_and_long_summary',
                   'keywords_and_countries_stored','embedding_stored','cost_reported']
        if not args.scheduled:
            before=len(fixture.callbacks); begin=time.monotonic()
            for endpoint in ('news','content'):
                result=session.post(base+'/trigger/'+endpoint,timeout=5)
                require(result.status_code==200 and result.json()['processed_count']==0,'重复执行又领取了已完成任务')
            require(len(fixture.callbacks)==before and time.monotonic()-begin<5,'空队列阻塞或重复回写')
            checks.append('empty_queue_and_repeat_run')
        if not args.live_models and not args.scheduled:
            for failure in ('http','business','json','dimension'):
                fixture.reset()
                if failure in ('http','business'):
                    fixture.fail_task='recTitle'; fixture.fail_kind=failure
                fixture.invalid_json=failure=='json'
                fixture.bad_dimension=failure=='dimension'
                result=session.post(base+'/trigger/news',timeout=30)
                (output/f'failure-{failure}.json').write_text(json.dumps({'http_status':result.status_code,'body':result.json()},ensure_ascii=False,indent=2))
                require(result.status_code==502,f'{failure} 被错误报告为成功')
                require(all('tag' not in r for r in fixture.records.values()),'失败任务被确认完成')
                checks.append('reject_'+failure+'_failure')
        report={'status':'passed','mode':'live-models/local-news' if args.live_models else 'offline',
                'server':args.server,'scheduled':args.scheduled,'checks':checks,
                'elapsed_seconds':round(time.monotonic()-started,2),
                'database':'local fixture JSON; real MySQL not tested',
                'model_quality':'schema and fixture expectations only; not a broad quality evaluation'}
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        fixture_server.shutdown()
        session.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-models',action='store_true',help='真实模型，会产生少量费用；新闻服务仍使用本地夹具')
    parser.add_argument('--scheduled',action='store_true',help='等待首次 10 秒自动任务，并检查下一轮间隔')
    parser.add_argument('--server',choices=['python','gunicorn'],default='python')
    parser.add_argument('--output',default=str(ROOT/'test-results'/'e2e'))
    args=parser.parse_args()
    try: run(args)
    except Exception as exc:
        print(f'E2E FAILED: {exc}',file=sys.stderr)
        raise SystemExit(1)
