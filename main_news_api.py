import os
import logging
import time, json
import threading
import requests
import traceback
import concurrent.futures
import numpy as np
from add_embedding_time import add_missing_embeddings

from flask import Flask, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler
from openai import OpenAI
from collections import defaultdict, Counter

from utils.parse import parse_json, count_words, calculate_abstract_length
from utils.prompt_templates import classify_prompt, base_content_prompt, short_content_prompt, country_normalization_map
from utils.database import VectorSearch
from config import LLM_CONFIG, NEWS_API_CONFIG, VECTOR_SEARCH_CONFIG, DOUBAO_CONFIG, EMBEDDING_CONFIG

# 初始化配置
app = Flask(__name__)
scheduler = BackgroundScheduler(daemon=True)

# 日志配置
# 配置日志同时输出到文件和控制台
formatter = logging.Formatter('[%(levelname)s-%(asctime)s] %(message)s')

# 创建文件处理器
file_handler = logging.FileHandler('news_analysis.log', encoding='utf-8')
file_handler.setFormatter(formatter)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# 获取根日志记录器并配置
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# 环境配置
OPENAI_KEY = LLM_CONFIG['API_KEY']
OPENAI_BASE_URL = LLM_CONFIG['BASE_URL']
BASE_MODEL_NAME = LLM_CONFIG['MODEL_NAME']

EMB_MODEL_NAME = EMBEDDING_CONFIG['MODEL_NAME']
EMB_URL = EMBEDDING_CONFIG['BASE_URL']
EMB_KEY = EMBEDDING_CONFIG['API_KEY']

DOUBAO_KEY = DOUBAO_CONFIG['API_KEY']
DOUBAO_URL = DOUBAO_CONFIG['BASE_URL']
DOUBAO_MODEL_NAME = DOUBAO_CONFIG['MODEL_NAME']

NEWS_API_BASE_URL = NEWS_API_CONFIG['BASE_URL']

THRESHOLD = VECTOR_SEARCH_CONFIG['THRESHOLD']
DIMENSION = VECTOR_SEARCH_CONFIG['DIMENSION']
true = True
false = False

# 初始化
faiss_matrix = VectorSearch(dim = DIMENSION, logger=logger)
LOAD_HOURS = VECTOR_SEARCH_CONFIG['LOAD_HOURS']
faiss_matrix.import_from_mysql(hours_limit=LOAD_HOURS)   # 修改导入函数，只导入最近小时的数据
logger.info(f"从数据库导入Embedding数量(最近{LOAD_HOURS}小时): {faiss_matrix.display_embedding_nums()}")

# 添加定时刷新任务
@scheduler.scheduled_job('interval', hours=1)
def refresh_faiss_matrix():
    """每小时刷新一次 faiss_matrix 并补充缺失的embedding"""
    try:
        # 补充缺失的embedding
        logger.info("开始补充缺失的embedding")
        result = add_missing_embeddings()
        if result:
            logger.info(f"补充embedding完成！成功: {result['success']}, 失败: {result['fail']}, 总数: {result['total']}")
        
        # 刷新faiss_matrix
        logger.info("开始刷新 faiss_matrix")
        global faiss_matrix
        new_matrix = VectorSearch(dim=DIMENSION, logger=logger)
        new_matrix.import_from_mysql(hours_limit=LOAD_HOURS)
        faiss_matrix = new_matrix
        logger.info(f"faiss_matrix 刷新完成，当前 Embedding 数量: {faiss_matrix.display_embedding_nums()}")
    except Exception as e:
        traceback.print_exc()
        logger.error(f"faiss_matrix 刷新失败: {str(e)}")
        traceback.print_exc()


def _call_openai_api(
    prompt, system_role, 
    api_key=OPENAI_KEY, 
    base_url=OPENAI_BASE_URL,
    model_name=BASE_MODEL_NAME
):
    """统一的OpenAI API调用方法"""
    # 初始化客户端
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = "未推理成功"
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt},
            ],
            response_format={'type': 'json_object'},
            temperature=0,
            stream=False
        )
        return response
    except Exception as e:
        traceback.print_exc()
        logger.error(f"OpenAI API调用失败: {str(e)}, {response}")
        return None
    
def _call_doubao_api_stream(
    prompt, system_role, 
    api_key=DOUBAO_KEY,
    base_url=DOUBAO_URL,
    model_name=DOUBAO_MODEL_NAME,
    stream_flag=False
):
    """统一的OpenAI API调用方法，返回流式响应"""
    # 初始化客户端
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        ans_str, response = "", "未推理成功"
        response = client.chat.completions.create(
            model = model_name,
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt},
            ],
            response_format={'type': 'json_object'},
            temperature=0.01,
            stream=stream_flag
        )
        if stream_flag:
            # 逐步接收数据
            usage_dict = {}
            for chunk in response:
                if chunk.choices[0].delta.content:
                    ans_str += chunk.choices[0].delta.content
                elif chunk.usage:   # 目前云无法返回这部分数据
                    usage_dict = chunk.usage.to_dict()
        else:
            # 一次性接收数据
            ans_str = response.choices[0].message.content
            usage_dict = response.usage.to_dict()
        return ans_str, usage_dict
    except Exception as e:
        logger.error(f"OpenAI API调用失败: {str(e)} - {response}")
        traceback.print_exc()
        return "", {}

def _call_embedding_api_doubao(
    text,
    api_key=EMB_KEY, base_url=EMB_URL,
    model=EMB_MODEL_NAME
):
    """豆包的embedding API调用方法"""
    try:
        client = OpenAI(api_key=EMB_KEY, base_url=EMB_URL)
        resp = "未推理成功"
        resp = client.embeddings.create(
            model=EMB_MODEL_NAME,
            input=[text],
            encoding_format="float"
        )
        vector = resp.data[0].embedding
        cost = resp.usage.to_dict()
        ans = np.array(vector).astype('float64').tolist()
        return True, ans, cost
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Embedding API调用失败: {str(e)} - {resp}")
        return False, e, 0


class NewsAnalyzer:
    def __init__(self):
        self.classify_prompt = classify_prompt
        self.base_content_prompt = base_content_prompt
        self.short_content_prompt = short_content_prompt

        self.model_costs = LLM_CONFIG['COSTS']
        self.embedding_id_dict = {}
        self.lock = threading.Lock()

    def analyze_news_list(self):
        """新闻列表分析核心逻辑"""
        try:
            logger.info("开始执行新闻列表分析任务")
            response = requests.get(f"{NEWS_API_BASE_URL}/getTask?taskname=recTitle&num=20&limit_time=10")
            
            if response.status_code != 200 or response.json().get('msg') != '获取完成':
                title_data = response.json()
                msg = f"获取新闻列表失败 - {title_data}"
                logger.error(msg)
                return False, 0
            
            title_data = response.json()['data']
            if not title_data:
                logger.warning("获取新闻列表为空")
                return False, 0 
            
            prompt_id_list = [item['id'] for item in title_data]
            prompt_title_list = [item['title'] for item in title_data]
            logger.info(f"[标题分析] 总计获取标题：{len(prompt_title_list)}")

            prompt = self.classify_prompt + str(prompt_title_list)
            llm_response, usage = _call_doubao_api_stream(
                prompt, 
                "你是新闻分类与判断专家", 
                model_name=BASE_MODEL_NAME)
            if len(llm_response) == 0:
                logger.warning("语言模型请求失败")
                return False,0

            # 记录使用量与成本
            model_cost = self.model_costs[BASE_MODEL_NAME]
            cost = (
                usage.get('prompt_tokens', 0) * model_cost['prompt'] +
                usage.get('completion_tokens', 0) * model_cost['completion'] +
                usage.get('prompt_cache_hit_tokens', 0) * model_cost.get('cache_prompt', 0)
            )

            flag, use_dict = parse_json(llm_response)
            if not flag:
                logger.error("分类结果解析失败")
                return False

            news_data = []
            for idx_str, ans_dict in use_dict.items():
                try:
                    idx = int(float(idx_str)) - 1
                    news_id = prompt_id_list[idx]
                    flag_value = ans_dict.get('flag', False)
                    news_classify = ans_dict.get('theme', [])
                    is_relevant = flag_value in (True, 'true', 'True')
                    if is_relevant:
                        emb_flag, embedding, emb_usage = _call_embedding_api_doubao(prompt_title_list[idx])
                        if emb_flag:
                            emb_usage = self.model_costs[EMB_MODEL_NAME]
                            emb_cost = emb_usage.get('prompt_tokens', 0) * model_cost['prompt']
                            add_flag = faiss_matrix.add_vector(embedding, thre=THRESHOLD)  # 相似度阈值，可以根据实际情况调整
                            tag = 1 if add_flag else -1
                            if add_flag:
                                self.embedding_id_dict[news_id] = embedding
                        else:
                            logging.warning(f"[analysis embedding error] {emb_flag} - {embedding}存在问题")
                            tag, emb_cost = 1, 0
                        # emb_flag, embedding = False, '未使用embedding'
                        # tag, emb_cost = 1, 0
                    else:
                        tag, emb_cost = 0, 0

                    news_single_dict = { 'id': news_id, 'tag': tag, 'cost': round(cost/len(use_dict) + emb_cost, 8) }
                    if tag:
                        if isinstance(news_classify, list):
                            news_single_dict['classify'] = news_classify[:1]
                        elif isinstance(news_classify, str):
                            news_single_dict['classify'] = [news_classify]
                        else:
                            news_single_dict['classify'] = []
                    news_data.append(news_single_dict)

                except Exception as e:
                    traceback.print_exc()
                    logger.warning(f"索引解析异常: {idx_str} - {str(e)}")

            recall_response = requests.post(
                f"{NEWS_API_BASE_URL}/recallTask",
                json={"taskname": "recTitle", "data": news_data}
            )
            logger.info(f"新闻列表分析完成，成功更新{recall_response.json().get('success_num', 0)}条 | 详情: {news_data} | 成本: {cost:.04f}")
            return True, cost
            
        except Exception as e:
            logger.error(f"新闻列表分析异常: {str(e)}\n{traceback.format_exc()}")
            traceback.print_exc()
            return False, 0

    def _get_content_list(self):
        """获取待处理内容列表（含重试机制）"""
        retry_count = 0
        while retry_count < 3:
            try:
                response = requests.get(
                    f"{NEWS_API_BASE_URL}/getTask",
                    params={
                        # "taskname": "genTranslate",
                        "taskname": "recCountry",
                        "num": 10,
                        "limit_time": 10
                    }
                )
                
                if response.json().get('num', 0) > 0:
                    return response.json()['data']
                
                if response.json().get('msg') == '当前已取净':
                    logger.info("内容列表已取净，等待重试...")
                    time.sleep(60)
                    retry_count += 1
                else:
                    return []
                    
            except Exception as e:
                logger.error(f"获取内容列表失败: {str(e)}")
                retry_count += 1
                time.sleep(30)
        
        return []

    def process_single_content(self, content_dict):
        """处理单个新闻内容，返回包含详细信息的字典"""
        content_id = content_dict.get('id', 'unknown')
        result_template = {
            'id': content_id,
            'status': 'success',
            'cost': 0.0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'translation_len': 0,
            'abstract_len': 0, 
            'original_len': len(content_dict.get('content', '')),
            'errors': [],
            'failed_callbacks': []
        }

        try:
            # API调用
            result_template['original_len'] = len(content_dict.get('content', ''))
            if not content_dict.get('content'):
                result_template.update({
                    'status': 'failed',
                    'errors': ['empty_content']
                })
                logger.warning(f"内容为空 ID:{content_id}")
                return result_template

            content_len = count_words(content_dict['content'])
            if content_len < 250:
                logger.info(f"开始处理内容 ID:{content_id}，原文长度:{content_len}")
                used_prompt = self.short_content_prompt
            else:
                min_abstract_len = calculate_abstract_length(content_len)
                max_abstract_len = min(min_abstract_len * 2, 500)
                logger.info(f"开始处理内容 ID:{content_id}，原文长度:{content_len}, 预期摘要长度:{min_abstract_len}-{max_abstract_len}")
                used_prompt = self.base_content_prompt.format(
                    min_abstract_len=min_abstract_len,
                    max_abstract_len=max_abstract_len
                )

            print(used_prompt)
            llm_response, usage = _call_doubao_api_stream(
                used_prompt + str(content_dict['content']),
                "你是新闻阅读与结构化整理专家", 
                model_name=DOUBAO_MODEL_NAME
            )
            if len(llm_response) == 0:
                result_template.update({
                    'status': 'failed',
                    'errors': ['api_call_failed']
                })
                return result_template

            # 记录使用量和成本
            model_cost = self.model_costs[DOUBAO_MODEL_NAME]
            cost = (
                usage.get('prompt_tokens', 0) * model_cost['prompt'] +
                usage.get('completion_tokens', 0) * model_cost['completion'] +
                usage.get('prompt_cache_hit_tokens', 0) * model_cost.get('cache_prompt', 0)
            )

            with self.lock:
                self.usage_records[content_id] = usage
                self.total_cost += cost

            result_template.update({
                'cost': round(cost, 8), 
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0)
            })

            # 解析结果
            flag, content_ans_dict = parse_json(llm_response)
            if not flag:
                result_template.update({
                    'status': 'failed',
                    'errors': ['json_parse_failed']
                })
                logger.error(f"JSON解析失败 ID:{content_id}")
                return result_template

            # 记录翻译长度
            translation = content_ans_dict.get('translate', '')
            if len(translation) == 0:
                logger.warning(f"翻译为空 ID:{content_id}, {content_ans_dict}")
            result_template['translation_len'] = len(translation)

            # 记录摘要长度
            abstract = content_ans_dict.get('abstract', translation)
            result_template['abstract_len'] = len(abstract)

            # 国家二次纠正
            country_list = content_ans_dict.get('country', [])
            norm_country_list = []
            for country in country_list:
                country = country.strip().split('(')[0].split('（')[0]
                country = country_normalization_map.get(country, country)
                norm_country_list.append(country)

            # 回调任务处理
            tasks = [
                ("genTranslate", translation),
                ("genAbstract", abstract),
                # ("genClassify", content_ans_dict.get('theme', []) + 
                #                 [f'公司-{tmp}' for tmp in content_ans_dict.get('company', '')] ),
                ("genKeyword", content_ans_dict.get('keywords', []) ),
                ("recCountry", norm_country_list ),
            ]

            # 判断是否回调embedding
            if content_id in self.embedding_id_dict:
                tasks.append(("genVec", self.embedding_id_dict[content_id]))
                del self.embedding_id_dict[content_id]

            callback_errors = []
            for taskname, result in tasks:
                try:
                    requests.post(
                        f"{NEWS_API_BASE_URL}/recallTask",
                        json={"taskname": taskname, 
                              "data": [{"id": content_id, "result": result, "cost": round(cost/len(tasks), 8)}]
                        },
                        timeout=10
                    )
                except Exception as e:
                    traceback.print_exc()
                    error_key = f"callback_{taskname}_failed"
                    callback_errors.append(error_key)
                    result_template['failed_callbacks'].append(taskname)
                    logger.warning(f"回调任务失败 {taskname}-{content_id}")


            if callback_errors:
                result_template.update({
                    'status': 'partial_success' if result_template['status'] == 'success' else 'failed',
                    'errors': result_template['errors'] + callback_errors
                })

            logger.info(
                f"内容处理成功 ID:{content_id} | 成本:{cost:.04f} | "
                f"原始长度:{content_len} | "
                f"翻译长度:{len(content_ans_dict.get('translate',''))} | "
                f"摘要长度:{len(content_ans_dict.get('abstract',''))}"
            )
            return result_template

        except Exception as e:
            traceback.print_exc()

            error_type = type(e).__name__
            trace = traceback.format_exc()
            with self.lock:
                self.usage_records[content_id] = {"error": error_type}

            result_template.update({
                'status': 'failed',
                'errors': [f'exception_{error_type}'],
                'error_details': trace
            })
            logger.error(f"处理失败 ID:{content_id} | 错误类型:{error_type} | 错误详情:{trace}")
            return result_template

    def batch_process_contents(self, batch_size=10):
        """批量处理新闻内容，返回增强的统计信息"""
        # 跑批单独初始化总成本
        self.usage_records = {}
        self.total_cost = 0.0

        try:
            start_time = time.time()
            content_list = self._get_content_list()
            if not content_list:
                return {
                    'status': 'success',
                    'message': '无任何文字',
                    'statistics': {}
                }

            logger.info(f"开始批量处理 {len(content_list)} 条内容")
            processing_results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_content = {
                    executor.submit(self.process_single_content, content): content
                    for content in content_list
                }
                
                for future in concurrent.futures.as_completed(future_to_content):
                    try:
                        result = future.result(timeout=300)
                        processing_results.append(result)
                    except Exception as e:
                        error_result = {
                            'id': future_to_content[future].get('id', 'unknown'),
                            'status': 'failed',
                            'errors': ['processing_timeout'],
                            'error_details': str(e),
                            'cost': 0
                        }
                        traceback.print_exc()
                        processing_results.append(error_result)

            # 生成统计信息
            stats = {
                'total': len(processing_results),
                'success': sum(1 for r in processing_results if r['status'] == 'success'),
                'partial_success': sum(1 for r in processing_results if r['status'] == 'partial_success'),
                'failed': sum(1 for r in processing_results if r['status'] == 'failed'),
                'total_cost': sum(r['cost'] for r in processing_results),
                # 'total_prompt_tokens': sum(r['prompt_tokens'] for r in processing_results),
                # 'total_completion_tokens': sum(r['completion_tokens'] for r in processing_results),
                'avg_processing_time': (time.time() - start_time) / len(processing_results),
                'common_errors': self._analyze_error_patterns(processing_results),
                'callback_failure_rates': self._calculate_callback_failure_rates(processing_results)
            }

            # 生成详细报告
            logger.info(f"\n{'='*30} 批量处理报告 {'='*30}"
                    f"\n[总量统计]"
                    f"\n• 总数: {stats['total']}"
                    f"\n• 成功: {stats['success']} ({stats['success']/stats['total']:.1%})"
                    f"\n• 部分成功: {stats['partial_success']} ({stats['partial_success']/stats['total']:.1%})"
                    f"\n• 失败: {stats['failed']} ({stats['failed']/stats['total']:.1%})"
                    f"\n• 总成本: {stats['total_cost']:.4f}"
                    f"\n• 平均处理时间: {stats['avg_processing_time']:.2f}s/item"
                    f"\n{'='*70}")

            return {
                'status': 'success',
                'processed_count': len(processing_results),
                'statistics': stats,
                'detailed_results': processing_results  # 返回前50条详情防止过大响应
            }

        except Exception as e:
            logger.error(f"批量处理异常: {str(e)}\n{traceback.format_exc()}")
            traceback.print_exc()
            return {
                'status': 'error',
                'error_type': type(e).__name__,
                'message': str(e)
            }
        
    # Helper methods
    def _analyze_error_patterns(self, results):
        error_counter = Counter()
        for result in results:
            error_counter.update(result['errors'])
        return dict(error_counter)

    def _calculate_callback_failure_rates(self, results):
        callback_stats = defaultdict(int)
        total = len(results)
        for result in results:
            for task in result.get('failed_callbacks', []):
                callback_stats[task] += 1
        return {k: f"{v/total:.1%}" for k, v in callback_stats.items()}


# 初始化处理器
news_analyzer = NewsAnalyzer()
# content_processor = ContentProcessor()

# 定时任务配置
@scheduler.scheduled_job('interval', minutes=10)  # 十分钟一次
def scheduled_news_analysis():
    for i in range(3):   # 执行3*20=60条新闻分析
        news_analyzer.analyze_news_list()

@scheduler.scheduled_job('interval', minutes=10)   # 十分钟一次
def scheduled_content_processing():
    for i in range(2):   # 执行2*10=20条新闻摘要、翻译等
        news_analyzer.batch_process_contents(batch_size=10)

# API接口
@app.route('/trigger/news', methods=['GET'])
def trigger_news_analysis():
    result_flag, cost = news_analyzer.analyze_news_list()
    return jsonify({
        "status": "success" if result_flag else "error",
        "message": "新闻列表分析已启动" if result_flag else "分析启动失败",
        "cost": round(cost, 8) if result_flag else 0.0
    })

@app.route('/trigger/content', methods=['GET'])
def trigger_content_processing():
    # result = content_processor.batch_process_contents()
    result = news_analyzer.batch_process_contents()
    return jsonify(result)

# scheduler.start()

if __name__ == '__main__':
    scheduler.start()
    try:
        app.run(host='0.0.0.0', port=2400, use_reloader=False)
    except KeyboardInterrupt:
        scheduler.shutdown()