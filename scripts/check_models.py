#!/usr/bin/env python3
"""Probe each configured model independently, using only synthetic text; no news/DB writes."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import LLM_CONFIG, DOUBAO_CONFIG, EMBEDDING_CONFIG
from utils.model_api import chat, embedding, validate_model_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='test-results/model-check.json')
    args=parser.parse_args()
    validate_model_config()
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    report=[]
    for label, settings in [('classify',LLM_CONFIG),('summary',DOUBAO_CONFIG),('embedding',EMBEDDING_CONFIG)]:
        begin=time.monotonic()
        row={'role':label,'model':settings['MODEL_NAME'],'base_url':settings['BASE_URL']}
        try:
            if label=='embedding':
                vector, usage=embedding('Saudi Arabia announced a defence procurement programme.')
                row.update(status='passed',dimension=len(vector),vector_preview=vector[:8],usage=usage)
            else:
                text,usage=chat('Return only this JSON object: {"ok":true}', 'Follow the output format precisely.',settings)
                parsed=json.loads(text.strip().removeprefix('```json').removesuffix('```').strip())
                if parsed != {'ok':True}: raise ValueError('Unexpected JSON output')
                row.update(status='passed',output=text,usage=usage)
        except Exception as exc:
            message=str(exc)
            for config in (LLM_CONFIG,DOUBAO_CONFIG,EMBEDDING_CONFIG):
                if config.get('API_KEY'):
                    message=message.replace(config['API_KEY'],'<redacted>')
            row.update(status='failed',error_type=type(exc).__name__,error=message)
        row['elapsed_seconds']=round(time.monotonic()-begin,2)
        report.append(row)
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(row,ensure_ascii=False),flush=True)
    return 0 if all(row['status']=='passed' for row in report) else 1


if __name__=='__main__':
    raise SystemExit(main())
