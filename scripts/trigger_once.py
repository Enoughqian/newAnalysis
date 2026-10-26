"""Explicit manual trigger; default one batch, bounded HTTP wait, nonzero exit on failure."""
import argparse
import json
import requests


def main(task):
    parser=argparse.ArgumentParser(description='手动触发处理，会领取新闻并回写结果')
    parser.add_argument('--base-url',default='http://127.0.0.1:2400')
    parser.add_argument('--repeat',type=int,default=1)
    args=parser.parse_args()
    if args.repeat<1: parser.error('--repeat 必须大于零')
    for _ in range(args.repeat):
        response=requests.post(args.base_url.rstrip('/')+'/trigger/'+task,timeout=(5,600))
        print(json.dumps(response.json(),ensure_ascii=False,indent=2))
        if response.status_code!=200 or response.json().get('status')!='success':
            return 1
        if not response.json().get('processed_count'):
            break
    return 0
