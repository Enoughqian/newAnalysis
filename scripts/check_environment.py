#!/usr/bin/env python3
"""Read-only business connectivity and vector-format check; never acquires news tasks."""
import json
from pathlib import Path
import socket
import sys
from urllib.parse import urlparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mysql.connector
from config import DB_CONFIG, NEWS_API_CONFIG, VECTOR_SEARCH_CONFIG


def main():
    report={}
    target=urlparse(NEWS_API_CONFIG['BASE_URL'])
    try:
        with socket.create_connection((target.hostname,target.port or 80),timeout=5):
            pass
        report['news_server']={'status':'tcp_reachable','note':'No getTask/recallTask request was sent'}
    except Exception as exc:
        report['news_server']={'status':'failed','error_type':type(exc).__name__}
    try:
        conn=mysql.connector.connect(**DB_CONFIG)
        try:
            cursor=conn.cursor()
            try:
                cursor.execute('SELECT OCTET_LENGTH(feature), COUNT(*) FROM news_detail '
                               'WHERE create_time >= DATE_SUB(NOW(), INTERVAL %s HOUR) '
                               'AND feature IS NOT NULL GROUP BY OCTET_LENGTH(feature)',
                               (VECTOR_SEARCH_CONFIG['LOAD_HOURS'],))
                report['mysql']={'status':'passed','vector_bytes_counts':cursor.fetchall(),
                                  'expected_float64_bytes':VECTOR_SEARCH_CONFIG['DIMENSION']*8}
            finally: cursor.close()
        finally: conn.close()
    except Exception as exc:
        report['mysql']={'status':'failed','error_type':type(exc).__name__,
                         'errno':getattr(exc,'errno',None)}
    path=Path(__file__).resolve().parents[1]/'test-results'/'environment-check.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 1 if any(v['status']=='failed' for v in report.values()) else 0


if __name__=='__main__': raise SystemExit(main())
