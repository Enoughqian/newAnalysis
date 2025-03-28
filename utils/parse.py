import re, json, ast

def safe_eval(data):
    try:
        if isinstance(data, (str, bytes)):
            result = eval(data)
        else:
            result = data
    except TypeError:
        result = data
    return result

def parse_json(text):
    """从文本中提取JSON并解析为Python字典"""
    # 尝试匹配带有 ```json``` 或其他大小写组合的标记
    match = re.search(r'```[jJ][sS][oO][nN]\n(.*?)\n```', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # 如果没有找到标记，尝试将整个输入文本作为JSON字符串来解析
        json_str = text.strip()

    # 去除可能存在的多余标记
    json_str = json_str.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

    try:
        # 尝试解析 JSON 字符串
        data_dict = json.loads(json_str)
        print(f"解析成功: {data_dict}")
        return True, data_dict
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        if isinstance(json_str, (str, bytes)):
            try:
                data_dict = ast.literal_eval(json_str)
                print(f"使用 ast.literal_eval 解析成功: {data_dict}")
                return True, data_dict
            except (ValueError, SyntaxError) as eval_error:
                print(f"ast.literal_eval 解析失败: {eval_error}")
                return False, f"{json_str} || 无法解析"
        else:
            print(f"返回无法解析：{json_str}")
            return False, f"{json_str} || 无法解析"

# 格式化耗时为可视化字符串
def format_timedelta(td):
    seconds = td.total_seconds()
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}小时{int(minutes)}分钟{int(seconds)}秒"

def remove_word(string, word_list=[]):
    for word in word_list:
        string = string.replace(word, '')
    return string #.strip()
    
def extract_numbers(s):
    numbers = re.findall(r'\d+', s)
    numbers = [int(num) for num in numbers]
    return numbers[:3]


def count_words(text: str) -> int:
    """计算英文文本的单词数量
    
    Args:
        text: 输入的英文文本
        
    Returns:
        单词数量
    """
    # 移除多余的空白字符并分割
    words = text.strip().split()
    return len(words)

def calculate_abstract_length(content_len: int) -> int:
    """
    根据原文长度计算合适的摘要长度
    Args:
        content_len: 原文长度
    Returns:
        abstract_len: 建议的摘要长度
    """
    if content_len < 400:
        return int(min(200, content_len/2))
    else:
        return int(min(300, content_len/3))