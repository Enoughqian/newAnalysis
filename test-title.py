import requests

def test_news_analysis():
    response = requests.get("http://localhost:2400/trigger/news")
    print(f"新闻分析接口响应: {response.status_code}")
    print(f"响应内容: {response.json()}")

for i in range(60):
    test_news_analysis()