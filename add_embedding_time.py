"""Backfill missing vectors or explicitly rebuild the recent history for a model migration."""
import argparse
import base64
import json
import os
from pathlib import Path
import mysql.connector
from config import DB_CONFIG, VECTOR_SEARCH_CONFIG, EMBEDDING_CONFIG
from utils.model_api import embedding, validate_model_config
from utils.news_api import callback


def add_missing_embeddings(hours=48, rebuild=False, dry_run=False, backup_path=None):
    if hours <= 0:
        raise ValueError('hours 必须大于零')
    if rebuild and not dry_run and not backup_path:
        raise ValueError('重建历史向量必须指定备份文件')
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        cursor = conn.cursor()
        try:
            query = ('SELECT title, unique_id, feature FROM news_detail '
                     'WHERE create_time >= DATE_SUB(NOW(), INTERVAL %s HOUR)')
            if not rebuild:
                query += ' AND (feature IS NULL OR LENGTH(feature) = 0)'
            cursor.execute(query, (hours,))
            rows = cursor.fetchall()
        finally:
            cursor.close()
    finally:
        conn.close()
    report = {'success': 0, 'fail': 0, 'total': len(rows), 'dry_run': dry_run,
              'model': EMBEDDING_CONFIG['MODEL_NAME'], 'dimension': VECTOR_SEARCH_CONFIG['DIMENSION'],
              'failed_ids': []}
    if dry_run:
        return report
    # Keep a source-level backup before any external writes. Never overwrite a previous backup.
    if backup_path:
        fd = os.open(Path(backup_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            for title, news_id, old_vector in rows:
                output.write(json.dumps({'id': news_id, 'title': title,
                    'feature_base64': base64.b64encode(old_vector).decode() if old_vector is not None else None},
                    ensure_ascii=False, default=str) + '\n')
    for title, news_id, _ in rows:
        try:
            vector, _usage = embedding(title)
            callback('genVec', [{'id': news_id, 'result': vector}])
            report['success'] += 1
        except Exception as exc:
            report['fail'] += 1
            report['failed_ids'].append(str(news_id))
            print(f'向量更新失败 ID={news_id}: {type(exc).__name__}')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=int, default=VECTOR_SEARCH_CONFIG['LOAD_HOURS'])
    parser.add_argument('--rebuild', action='store_true', help='包括已有向量；用于更换模型/维度/指令')
    parser.add_argument('--apply', action='store_true', help='实际调用模型并回写；默认只统计')
    parser.add_argument('--backup', help='重建前保存现有向量，要求文件不存在')
    args = parser.parse_args()
    validate_model_config()
    report = add_missing_embeddings(args.hours, args.rebuild, not args.apply, args.backup)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if report['fail'] else 0)
