import requests

def test_content_analysis():
    response = requests.get("http://localhost:2400/trigger/content")
    print(f"内容分析接口响应: {response.status_code}")
    print("响应内容")
    print(response.json())

for i in range(30):
    test_content_analysis()