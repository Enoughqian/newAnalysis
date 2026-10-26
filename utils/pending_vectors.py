"""Durable hand-off when recTitle precedes the crawler creating news_detail."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from config import EMBEDDING_CONFIG, VECTOR_SEARCH_CONFIG


class PendingVectors:
    def __init__(self, path=None):
        self.path=Path(path or os.getenv('PENDING_VECTOR_DB',str(Path(__file__).resolve().parents[1]/'.cache'/'pending-vectors.sqlite3')))
        identity={k:EMBEDDING_CONFIG.get(k) for k in ('MODEL_NAME','BASE_URL','INSTRUCTIONS','API_TYPE')}
        identity['dimension']=VECTOR_SEARCH_CONFIG['DIMENSION']
        self.namespace=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        conn=sqlite3.connect(self.path,timeout=10)
        try:
            with conn:
                conn.execute('CREATE TABLE IF NOT EXISTS vectors (namespace TEXT, news_id TEXT, vector TEXT, confirmed INTEGER, created REAL, PRIMARY KEY(namespace,news_id))')
                yield conn
        finally:
            conn.close()

    def save(self, rows):
        with self.connect() as conn:
            conn.executemany('INSERT OR REPLACE INTO vectors VALUES (?,?,?,?,?)',
                [(self.namespace,str(row['id']),json.dumps(row['result']),0,time.time()) for row in rows])

    def confirm(self, ids):
        with self.connect() as conn:
            conn.executemany('UPDATE vectors SET confirmed=1 WHERE namespace=? AND news_id=?',[(self.namespace,str(i)) for i in ids])

    def get(self, ident):
        with self.connect() as conn:
            row=conn.execute('SELECT vector FROM vectors WHERE namespace=? AND news_id=? AND confirmed=1',
                              (self.namespace,str(ident))).fetchone()
        return json.loads(row[0]) if row else None

    def delete(self, ident):
        with self.connect() as conn:
            conn.execute('DELETE FROM vectors WHERE namespace=? AND news_id=?',(self.namespace,str(ident)))

    def recent(self, hours):
        with self.connect() as conn:
            rows=conn.execute('SELECT news_id,vector FROM vectors WHERE namespace=? AND confirmed=1 AND created>=?',
                              (self.namespace,time.time()-hours*3600)).fetchall()
        return [(ident,json.loads(vector)) for ident,vector in rows]
