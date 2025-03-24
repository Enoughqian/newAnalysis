import os
import requests
import json
import numpy as np
import mysql.connector
from datetime import datetime, timedelta
from tqdm import tqdm
from openai import OpenAI

from config import EMBEDDING_CONFIG, NEWS_API_CONFIG, DB_CONFIG

EMB_URL = EMBEDDING_CONFIG['BASE_URL']
EMB_KEY = EMBEDDING_CONFIG['API_KEY']
EMB_MODEL_NAME = EMBEDDING_CONFIG['MODEL_NAME']

recall_url = NEWS_API_CONFIG['RECALL_URL']


def _call_embedding_api_doubao(
    text,
    api_key=EMB_KEY, base_url=EMB_URL,
    model=EMB_MODEL_NAME
):
    """豆包的embedding API调用方法"""
    try:
        client = OpenAI(api_key=EMB_KEY, base_url=EMB_URL)

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
        logger.error(f"Embedding API调用失败: {str(e)}")
        return False, e, 0

def add_missing_embeddings(hours=48):
    """补充最近指定小时数内缺失的新闻embedding"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 查询最近48小时内且feature为空的新闻
        query = """
            SELECT title, unique_id 
            FROM news_detail 
            WHERE update_time >= DATE_SUB(NOW(), INTERVAL %s HOUR)
            AND (feature IS NULL OR LENGTH(feature) = 0)
        """
        cursor.execute(query, (hours,))
        rows = cursor.fetchall()
        
        success_count = 0
        fail_count = 0
        
        for row in tqdm(rows, desc="处理缺失embedding的新闻"):
            title, title_id = row
            try:
                # 调用embedding API
                flag, ans, cost = _call_embedding_api_doubao(title)
                
                # 调用回调接口更新向量
                data = {
                    "taskname": "genVec",
                    "data": [
                        {"id": title_id, "result": ans},
                    ]
                }
                call_response = requests.post(recall_url, json=data)
                
                if call_response.json().get('err_code') == 0:
                    success_count += 1
                else:
                    fail_count += 1
                    
            except Exception as e:
                print(f"处理新闻ID {title_id} 失败: {str(e)}")
                fail_count += 1
                continue
                
        return {
            "success": success_count,
            "fail": fail_count,
            "total": len(rows)
        }
                
    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    result = add_missing_embeddings()
    if result:
        print(f"处理完成！成功: {result['success']}, 失败: {result['fail']}, 总数: {result['total']}")