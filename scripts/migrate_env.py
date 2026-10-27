#!/usr/bin/env python3
"""Copy literal credentials from a legacy config.py into an ignored .env; never print values."""
import argparse
import ast
import os
from pathlib import Path
import re
import shlex


def extract(source):
    blocks = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            values = {}
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                    values[key.value] = value.value
            blocks[node.targets[0].id] = values
    output = {}
    keys = [blocks.get(name, {}).get('API_KEY') for name in ('LLM_CONFIG', 'DOUBAO_CONFIG', 'EMBEDDING_CONFIG')]
    if all(keys) and len(set(keys)) == 1:
        output['ARK_API_KEY'] = keys[0]
    else:
        for prefix, key in zip(('CLASSIFY', 'SUMMARY', 'EMBEDDING'), keys):
            if key: output[prefix + '_API_KEY'] = key
    password = blocks.get('DB_CONFIG', {}).get('password')
    if password: output['DB_PASSWORD'] = password
    for field in ('host', 'user', 'database', 'port'):
        value = blocks.get('DB_CONFIG', {}).get(field)
        if value is not None: output['DB_' + field.upper()] = value
    news_url = blocks.get('NEWS_API_CONFIG', {}).get('BASE_URL')
    if news_url: output['NEWS_API_BASE_URL'] = news_url
    return output


def migrate(source, destination):
    values = extract(Path(source).read_text())
    path = Path(destination)
    existing = path.read_text() if path.exists() else ''
    populated = set()
    for line in existing.splitlines():
        match = re.match(r'(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)', line.strip())
        if match and match[2].strip().strip('\'"'):
            populated.add(match[1])
    additions = {k:v for k,v in values.items() if k not in populated}
    if additions:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, 'a') as output:
            if existing and not existing.endswith('\n'): output.write('\n')
            for key,value in additions.items(): output.write(key + '=' + shlex.quote(str(value)) + '\n')
        os.chmod(path, 0o600)
    return list(additions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', default='.env')
    args = parser.parse_args()
    names = migrate(args.source, args.output)
    print('已保存变量（不显示值）：' + (', '.join(names) if names else '无新增；已有非空配置保持不变'))
