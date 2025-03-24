import os
import requests
import json
import faiss
import numpy as np
import mysql.connector
from tqdm import tqdm

from config import EMBEDDING_CONFIG, NEWS_API_CONFIG, DB_CONFIG

url = EMBEDDING_CONFIG['BASE_URL']
api_key = EMBEDDING_CONFIG['API_KEY']
MODEL_NAME = EMBEDDING_CONFIG['MODEL_NAME']

recall_url = NEWS_API_CONFIG['RECALL_URL']

conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()
cursor.execute("SELECT title, feature, unique_id FROM news_detail")

for row in tqdm(cursor):
    title, feature, title_id = row
    try:
        tmp_feature = np.frombuffer(feature, dtype=np.float64).astype('float64')
    except:
        tmp_feature = []
    if feature is None:# or len(tmp_feature)!=1024:
        payload = json.dumps({"model": MODEL_NAME, "input": title})
        
        response = requests.request("POST", url, 
                                  headers={
                                      'Authorization': f'Bearer {api_key}',
                                      'Content-Type': 'application/json'
                                  }, 
                                  data=payload).json()
        vector = response['data'][0]['embedding']
        ans = np.array(vector).astype('float64').tolist()
        data = {
            "taskname": "genVec",
            "data": [
                {"id": title_id, "result": ans},
            ]
        }
        call_response = requests.post(recall_url, json=data)
        print(call_response.json())