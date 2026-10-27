#!/usr/bin/env python3
"""Generate auditable real-model outputs from synthetic news. No business DB or callbacks."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import LLM_CONFIG, DOUBAO_CONFIG
from utils.model_api import chat, estimate_cost
from utils.parse import count_words, calculate_abstract_length
from utils.prompt_templates import classify_prompt, base_content_prompt, short_content_prompt
from main_news_api import validate_classification, validate_summary
from e2e import ROWS


def main():
    output=Path(__file__).resolve().parents[1]/'test-results'/'real-text-samples'
    output.mkdir(parents=True,exist_ok=True)
    os.environ['MODEL_AUDIT_FILE']=str(output/'models.jsonl')
    (output/'models.jsonl').write_text('')
    result={'mode':'real-models, synthetic inputs, no business writes', 'inputs':ROWS,
            'embedding':'not exercised by this text-only sample script; see e2e-live-models/report.json', 'results':[]}
    def save():
        (output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    save()
    text,usage=chat(classify_prompt+json.dumps([r['title'] for r in ROWS],ensure_ascii=False),
                    '你是新闻分类与判断专家',LLM_CONFIG)
    parsed=validate_classification(text,len(ROWS))
    result['classification']={'model':LLM_CONFIG['MODEL_NAME'],'output':parsed,'usage':usage,
                              'estimated_cost':estimate_cost(LLM_CONFIG['MODEL_NAME'],usage)}
    save()
    for row in (ROWS[0],ROWS[3]):
        n=count_words(row['content'])
        minimum=calculate_abstract_length(n)
        prompt=short_content_prompt if n<250 else base_content_prompt.format(min_abstract_len=minimum,max_abstract_len=min(2*minimum,300))
        text,usage=chat(prompt+row['content'],'你是新闻阅读与结构化整理专家',DOUBAO_CONFIG)
        parsed=validate_summary(text)
        result['results'].append({'id':row['id'],'model':DOUBAO_CONFIG['MODEL_NAME'],'original_word_count':n,
                                  'output':parsed,'usage':usage,'estimated_cost':estimate_cost(DOUBAO_CONFIG['MODEL_NAME'],usage),
                                  'title_length':len(parsed['title']),'abstract_length':len(parsed['abstract'])})
        save()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
