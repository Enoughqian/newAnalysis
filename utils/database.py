import logging
import faiss
import numpy as np
import mysql.connector
from config import DB_CONFIG, VECTOR_SEARCH_CONFIG
from utils.model_api import validate_vector


class VectorSearch:
    """Caller serializes mutations. A staged copy is committed only after callbacks succeed."""
    def __init__(self, dim, logger=None):
        self.dim = dim
        self.logger = logger or logging.getLogger(__name__)
        self.index = faiss.IndexFlatIP(dim)
        self.news_ids = []

    def clone(self):
        other = VectorSearch(self.dim, self.logger)
        other.index = faiss.clone_index(self.index)
        other.news_ids = list(self.news_ids)
        return other

    def add_vector(self, vector, thre=VECTOR_SEARCH_CONFIG['THRESHOLD'], news_id=None):
        validate_vector(vector, self.dim)
        # Retrying a callback for the same article must not mark it as its own duplicate.
        key = str(news_id) if news_id is not None else None
        if key is not None and key in self.news_ids:
            return True
        values = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if not np.isfinite(values).all():
            raise ValueError('向量超出 float32 范围')
        faiss.normalize_L2(values)
        if self.index.ntotal:
            scores, _ = self.index.search(values, 1)
            if scores[0][0] > thre:
                return False
        self.index.add(values)
        self.news_ids.append(key)
        return True

    def import_from_mysql(self, hours_limit=None):
        conn = mysql.connector.connect(**DB_CONFIG)
        try:
            cursor = conn.cursor()
            try:
                query = 'SELECT unique_id, feature FROM news_detail WHERE feature IS NOT NULL AND LENGTH(feature) > 0'
                params = ()
                if hours_limit is not None:
                    query += ' AND create_time >= DATE_SUB(NOW(), INTERVAL %s HOUR)'
                    params = (hours_limit,)
                cursor.execute(query, params)
                rows = cursor.fetchall()
            finally:
                cursor.close()
            # Reject incompatible history instead of silently using an incomplete index.
            for news_id, blob in rows:
                vector = np.frombuffer(blob, dtype=np.float64).tolist()
                self.add_vector(vector, news_id=news_id)
            self.logger.info('历史向量导入完成: %s 条记录，%s 条去重索引', len(rows), self.index.ntotal)
            return len(rows)
        finally:
            conn.close()

    def delete_all_embeddings(self):
        self.index.reset()
        self.news_ids.clear()

    def display_embedding_nums(self):
        return self.index.ntotal
