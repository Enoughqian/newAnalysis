import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('ARK_API_KEY', 'offline-test-key')
os.environ['DB_ENABLED'] = '0'
os.environ['SCHEDULER_ENABLED'] = '0'
import logging
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import requests
import main_news_api as service
from config import LLM_CONFIG, DOUBAO_CONFIG, EMBEDDING_CONFIG, RUNTIME_CONFIG
from utils import model_api, news_api
from utils.database import VectorSearch
from add_embedding_time import add_missing_embeddings


def response(body, status=200):
    result = Mock(status_code=status, ok=status < 400)
    result.json.return_value = body
    if status >= 400:
        result.raise_for_status.side_effect = requests.HTTPError(str(status))
    return result


class RegressionTests(unittest.TestCase):
    def setUp(self):
        pending_patch = patch.object(service, 'pending_vectors')
        pending = pending_patch.start(); pending.get.return_value = None; pending.recent.return_value = []
        self.addCleanup(pending_patch.stop)
        service.faiss_matrix = VectorSearch(2048)
        service.ready = False
        service.initialize_service()
        self.client = service.app.test_client()
        self.vector = [1.0] + [0.0] * 2047
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_classification_uses_own_config_and_books_embedding_cost(self):
        with patch.object(service, 'fetch_tasks', return_value=[{'id': 'one', 'title': 'Saudi defence'}]), \
             patch.object(service, 'chat', return_value=('{"1":{"flag":true,"theme":"军事"}}', {'prompt_tokens':100})) as chat, \
             patch.object(service, 'embedding', return_value=(self.vector, {'prompt_tokens':100})), \
             patch.object(service, 'callback') as cb:
            result = self.client.post('/trigger/news')
        self.assertEqual(result.status_code, 200)
        self.assertIs(chat.call_args.args[2], LLM_CONFIG)
        self.assertEqual([c.args[0] for c in cb.call_args_list], ['recTitle'])
        expected = model_api.estimate_cost(LLM_CONFIG['MODEL_NAME'], {'prompt_tokens':100}) + model_api.estimate_cost(EMBEDDING_CONFIG['MODEL_NAME'], {'prompt_tokens':100})
        self.assertAlmostEqual(result.json['cost'], expected)
        service.pending_vectors.save.assert_called_once()
        service.pending_vectors.confirm.assert_called_once_with(['one'])
        self.assertEqual(service.faiss_matrix.index.ntotal, 1)

    def test_bad_json_or_missing_rows_does_not_callback(self):
        for value in ('garbage', '[]', '{}', '{"2":{"flag":false}}', '{"1":{"flag":"true"}}'):
            with self.subTest(value=value), \
                 patch.object(service, 'fetch_tasks', return_value=[{'id': 'one','title':'Title'}]), \
                 patch.object(service, 'chat', return_value=(value, {})), patch.object(service, 'callback') as cb:
                result = self.client.post('/trigger/news')
                self.assertEqual(result.status_code, 502)
                cb.assert_not_called()

    def test_invalid_embedding_never_marks_article_as_relevant(self):
        with patch.object(service, 'fetch_tasks', return_value=[{'id': 'one','title':'Title'}]), \
             patch.object(service, 'chat', return_value=('{"1":{"flag":true,"theme":"军事"}}', {})), \
             patch.object(service, 'embedding', side_effect=ValueError('bad dimension')), \
             patch.object(service, 'callback') as cb:
            self.assertEqual(self.client.post('/trigger/news').status_code, 502)
            cb.assert_not_called()

    def test_callback_failure_does_not_commit_index(self):
        with patch.object(service, 'fetch_tasks', return_value=[{'id':'one','title':'Title'}]), \
             patch.object(service, 'chat', return_value=('{"1":{"flag":true,"theme":"军事"}}',{})), \
             patch.object(service, 'embedding', return_value=(self.vector, {})), \
             patch.object(service, 'callback', side_effect=ValueError('failed')):
            self.assertEqual(self.client.post('/trigger/news').status_code, 502)
            self.assertEqual(service.faiss_matrix.index.ntotal, 0)

    def test_callback_http_business_and_partial_failures(self):
        cases = [(500, {'err_code':0}), (200, {'err_code':1}),
                 (200, {'err_code':0,'success_num':0}), (200, {}), (200, [])]
        for status, body in cases:
            with self.subTest(status=status,body=body), patch.object(news_api.requests,'post',return_value=response(body,status)):
                with self.assertRaises(Exception):
                    news_api.callback('recTitle',[{'id':'one'}])
        for body in ({'err_code':0},{'success_num':1},{'err_code':0,'success_num':1}):
            with patch.object(news_api.requests,'post',return_value=response(body)):
                news_api.callback('recTitle',[{'id':'one'}])

    def test_content_failure_does_not_ack_queue(self):
        summary = '{"title":"标题。","abstract":"正文","keywords":["关键词"],"country":["美国"]}'
        with patch.object(service,'chat',return_value=(summary,{})) as chat, \
             patch.object(service,'callback',side_effect=ValueError('rejected')) as cb:
            result = service.news_analyzer.process_single_content({'id':'one','content':'News content'})
        self.assertIs(chat.call_args.args[2], DOUBAO_CONFIG)
        self.assertEqual(result['status'],'partial_success')
        self.assertEqual(result['failed_callbacks'],['genAbstract'])
        self.assertEqual(cb.call_count,1)

    def test_content_batch_propagates_failure(self):
        with patch.object(service,'fetch_tasks',return_value=[{'id':'one','content':''}]):
            result = self.client.post('/trigger/content')
        self.assertEqual(result.status_code,502)
        self.assertEqual(result.json['statistics']['failed'],1)

    def test_empty_queue_has_no_sleep(self):
        with patch.object(news_api.requests,'get',return_value=response({'msg':'当前已取净','num':0})), \
             patch.object(news_api.time,'sleep') as sleep:
            self.assertEqual(news_api.fetch_tasks('recCountry',10),[])
            sleep.assert_not_called()

    def test_transport_timeout_is_not_retried_for_leased_task(self):
        with patch.object(news_api.requests,'get',side_effect=requests.Timeout()) as get:
            with self.assertRaises(requests.Timeout):
                news_api.fetch_tasks('recTitle',20)
            self.assertEqual(get.call_count,1)
            self.assertIn('timeout',get.call_args.kwargs)

    def test_multimodal_payload_and_response(self):
        with patch.object(model_api.requests,'post',return_value=response({'data':{'embedding':self.vector},'usage':{'prompt_tokens':3}})) as post:
            vector, usage = model_api.embedding('Title')
        self.assertEqual(len(vector),2048)
        self.assertEqual(usage['prompt_tokens'],3)
        self.assertTrue(post.call_args.args[0].endswith('/embeddings/multimodal'))
        self.assertEqual(post.call_args.kwargs['json']['input'],[{'type':'text','text':'Title'}])
        self.assertEqual(post.call_args.kwargs['json']['dimensions'],2048)

    def test_vector_validation_and_same_id_retry(self):
        search = VectorSearch(2)
        for vector in ([0,0],[1],[float('nan'),1],[float('inf'),1],['x',1]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                search.add_vector(vector)
        self.assertTrue(search.add_vector([1,0], news_id='a'))
        self.assertTrue(search.add_vector([1,0], news_id='a'))
        self.assertFalse(search.add_vector([1,0], news_id='b'))
        self.assertEqual(search.index.ntotal,1)

    def test_concurrent_trigger_returns_busy(self):
        with service.processing_lock:
            self.assertEqual(self.client.post('/trigger/news').status_code,409)
            self.assertEqual(self.client.get('/healthz').status_code,200)

    def test_schedule_first_10_seconds_and_interval_600(self):
        fake = Mock(running=False)
        started = datetime.now(timezone.utc)
        with patch.object(service,'scheduler',fake), patch.dict(RUNTIME_CONFIG,{'SCHEDULER_ENABLED':True}):
            service.start_scheduler()
        args = fake.add_job.call_args.kwargs
        self.assertEqual(args['seconds'],600)
        self.assertAlmostEqual((args['next_run_time']-started).total_seconds(),10,delta=1)
        self.assertEqual(args['max_instances'],1)

    def test_pipeline_orders_classification_before_content(self):
        calls=[]
        with patch.object(service.news_analyzer,'analyze_news_list',side_effect=lambda: calls.append('title') or {'status':'success','processed_count':0}), \
             patch.object(service.news_analyzer,'batch_process_contents',side_effect=lambda: calls.append('content') or {'status':'success','processed_count':0}):
            service.run_pipeline()
        self.assertEqual(calls,['title','content'])

    def test_empty_database_import(self):
        conn=Mock(); conn.cursor.return_value.fetchall.return_value=[]
        with patch('utils.database.mysql.connector.connect',return_value=conn):
            self.assertEqual(VectorSearch(2048).import_from_mysql(48),0)
        conn.close.assert_called_once()

    def test_backfill_failure_never_sends_invalid_vector(self):
        conn=Mock(); conn.cursor.return_value.fetchall.return_value=[('Title','one',None)]
        with patch('add_embedding_time.mysql.connector.connect',return_value=conn), \
             patch('add_embedding_time.embedding',side_effect=ValueError('failed')), \
             patch('add_embedding_time.callback') as cb:
            report=add_missing_embeddings()
        self.assertEqual(report['fail'],1)
        cb.assert_not_called()

    def test_summary_length_does_not_block_or_truncate(self):
        import json
        sample={'title':'字'*30,'abstract':'字'*600,'keywords':['测试'],'country':[]}
        self.assertEqual(service.validate_summary(json.dumps(sample),max_abstract_len=100),sample)

    def test_cache_cost_is_not_double_counted(self):
        with patch.dict(LLM_CONFIG['COSTS'],{'test':{'prompt':2,'completion':3,'cache_prompt':1}}):
            self.assertEqual(model_api.estimate_cost('test',{'prompt_tokens':100,'completion_tokens':10,
                                                          'prompt_tokens_details':{'cached_tokens':40}}),190)



class PendingVectorTests(unittest.TestCase):
    def test_confirmed_only_and_durable_handoff(self):
        import tempfile
        from pathlib import Path
        from utils.pending_vectors import PendingVectors
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'vectors.sqlite3'
            first=PendingVectors(path)
            first.save([{'id':1,'result':[1.0,0.0]}])
            self.assertIsNone(first.get(1))
            first.confirm([1])
            second=PendingVectors(path)
            self.assertEqual(second.get(1),[1.0,0.0])
            self.assertEqual(second.recent(48),[('1',[1.0,0.0])])
            second.delete(1)
            self.assertIsNone(first.get(1))

class OnlineContractTests(unittest.TestCase):
    def test_empty_country_uses_empty_value_not_fabricated_country(self):
        with patch.object(news_api.requests,'post',return_value=response({'err_code':0,'success_num':1})) as post:
            news_api.callback('recCountry',[{'id':1,'result':[]}])
        self.assertEqual(post.call_args.kwargs['json']['data'][0]['result'],[''])

    def test_empty_title_queue_with_blank_message(self):
        with patch.object(news_api.requests,'get',return_value=response({'data':[],'err_code':0,'num':0,'msg':''})):
            self.assertEqual(news_api.fetch_tasks('recTitle',1),[])


class EnvMigrationTests(unittest.TestCase):
    def test_legacy_credentials_are_migrated_without_overwriting_existing(self):
        import tempfile
        from scripts.migrate_env import migrate
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'legacy.py'; target=Path(directory)/'.env'
            source.write_text("LLM_CONFIG={'API_KEY':'test-key'}\nDOUBAO_CONFIG={'API_KEY':'test-key'}\nEMBEDDING_CONFIG={'API_KEY':'test-key'}\nDB_CONFIG={'password':'test-password','host':'db.example'}\n")
            target.write_text("ARK_API_KEY='existing-test-key'\nDB_PASSWORD=''\n")
            names=migrate(source,target)
            self.assertNotIn('ARK_API_KEY',names)
            self.assertIn('DB_PASSWORD',names)
            self.assertIn('existing-test-key',target.read_text())
            self.assertEqual(target.stat().st_mode & 0o777,0o600)


if __name__ == '__main__':
    unittest.main()
