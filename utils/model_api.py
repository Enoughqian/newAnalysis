"""Shared Ark adapters. No database access or network I/O at import time."""
import math
import os
import json
import threading
from datetime import datetime, timezone

_audit_lock = threading.Lock()

def audit_record(kind, settings, **fields):
    path = os.getenv("MODEL_AUDIT_FILE")
    if path:
        record = {"time": datetime.now(timezone.utc).isoformat(), "kind": kind,
                  "model": settings["MODEL_NAME"], **fields}
        with _audit_lock:
            with open(path, "a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

import requests
from openai import OpenAI
from config import LLM_CONFIG, EMBEDDING_CONFIG, VECTOR_SEARCH_CONFIG, RUNTIME_CONFIG


def chat(prompt, system_role, settings):
    extra = {}
    if settings.get('REASONING_EFFORT'):
        extra['reasoning_effort'] = settings['REASONING_EFFORT']
    limits = {}
    if os.getenv('MODEL_MAX_TOKENS'):
        limits['max_tokens'] = int(os.environ['MODEL_MAX_TOKENS'])
    with OpenAI(api_key=settings['API_KEY'], base_url=settings['BASE_URL'],
                timeout=RUNTIME_CONFIG['MODEL_TIMEOUT'], max_retries=0) as client:
        response = client.chat.completions.create(
            model=settings['MODEL_NAME'],
            messages=[{'role': 'system', 'content': system_role},
                      {'role': 'user', 'content': prompt}],
            stream=False, extra_body=extra, **limits,
        )
    usage = response.usage.model_dump() if response.usage else {}
    content = response.choices[0].message.content if response.choices else None
    audit_record('chat', settings, input=prompt, system_role=system_role, output=content,
                 usage=usage, response_id=response.id)
    error = None
    if not content:
        error = ValueError('模型未返回正文')
    elif response.choices[0].finish_reason not in (None, 'stop'):
        error = ValueError('模型输出未正常完成: ' + str(response.choices[0].finish_reason))
    if error:
        error.usage = usage
        raise error
    return content, usage


def validate_vector(vector, dimension=None):
    dimension = dimension or VECTOR_SEARCH_CONFIG['DIMENSION']
    if not isinstance(vector, list) or len(vector) != dimension:
        raise ValueError(f'向量维度错误，要求 {dimension}')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector):
        raise ValueError('向量包含非数值或非有限值')
    if not any(x != 0 for x in vector):
        raise ValueError('向量不可全为零')
    return vector


def embedding(text, settings=None):
    settings = settings or EMBEDDING_CONFIG
    if not isinstance(text, str) or not text.strip():
        raise ValueError('向量输入为空')
    if settings['API_TYPE'] == 'multimodal':
        body = {'model': settings['MODEL_NAME'], 'input': [{'type': 'text', 'text': text}],
                'encoding_format': 'float', 'dimensions': VECTOR_SEARCH_CONFIG['DIMENSION']}
        if settings.get('INSTRUCTIONS'):
            body['instructions'] = settings['INSTRUCTIONS']
        response = requests.post(settings['BASE_URL'] + '/embeddings/multimodal',
                                 headers={'Authorization': 'Bearer ' + settings['API_KEY']},
                                 json=body, timeout=(10, RUNTIME_CONFIG['MODEL_TIMEOUT']))
        if not response.ok:
            try:
                error = response.json().get('error', {})
                code = error.get('code', 'unknown')
            except (ValueError, AttributeError):
                code = 'non_json_response'
            raise RuntimeError(f'Embedding HTTP {response.status_code}: {code}')
        data = response.json()
        vector = data['data']['embedding']
        usage = data.get('usage', {})
    elif settings['API_TYPE'] == 'text':
        with OpenAI(api_key=settings['API_KEY'], base_url=settings['BASE_URL'],
                    timeout=RUNTIME_CONFIG['MODEL_TIMEOUT'], max_retries=0) as client:
            response = client.embeddings.create(model=settings['MODEL_NAME'], input=[text],
                                                encoding_format='float')
        vector = response.data[0].embedding
        usage = response.usage.model_dump() if response.usage else {}
    else:
        raise ValueError('EMBEDDING_API_TYPE 必须是 multimodal 或 text')
    validate_vector(vector)
    audit_record('embedding', settings, input=text, dimension=len(vector), vector_preview=vector[:8], usage=usage)
    return vector, usage


def estimate_cost(model, usage):
    """Estimated CNY; configured prices are per token. Cached tokens are not charged twice."""
    rates = LLM_CONFIG['COSTS'].get(model)
    if rates is None:
        raise ValueError(f'请在 LLM_CONFIG.COSTS 中配置模型单价: {model}')
    prompt = usage.get('prompt_tokens', 0) or 0
    cached = usage.get('prompt_cache_hit_tokens',
                       (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)) or 0
    cached = max(0, min(prompt, cached))
    return ((prompt - cached) * rates['prompt'] +
            cached * rates.get('cache_prompt', rates['prompt']) +
            (usage.get('completion_tokens', 0) or 0) * rates.get('completion', 0))


def validate_model_config():
    from config import DOUBAO_CONFIG
    for settings in (LLM_CONFIG, DOUBAO_CONFIG, EMBEDDING_CONFIG):
        if not settings['API_KEY'] or not settings['MODEL_NAME']:
            raise ValueError('模型名称和 API Key 不可为空')
        estimate_cost(settings['MODEL_NAME'], {})
    if EMBEDDING_CONFIG['API_TYPE'] not in ('multimodal', 'text'):
        raise ValueError('未知向量接口类型')
    if EMBEDDING_CONFIG['MODEL_NAME'].startswith('doubao-embedding-vision'):
        if EMBEDDING_CONFIG['API_TYPE'] != 'multimodal' or VECTOR_SEARCH_CONFIG['DIMENSION'] not in (1024, 2048):
            raise ValueError('doubao-embedding-vision 要求 multimodal 接口和 1024/2048 维')
