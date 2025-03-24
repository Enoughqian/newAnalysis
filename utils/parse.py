import re, json

def safe_eval(data):
    try:
        if isinstance(data, (str, bytes)):
            result = eval(data)
        else:
            result = data
    except TypeError:
        result = data
    return result
'''
def parse_json(input_json):
    if isinstance(input_json, dict):
        return True, input_json

    try:
        # Attempt to parse the JSON input
        input_json = input_json.split("```json")[-1].split("```")[0]
        data_dict = json.loads(input_json)
        # Convert lists to newline-separated strings
        # for key, value in data_dict.items():
        #     if isinstance(value, list):
        #         data_dict[key] = '\n'.join(value)
        # 打印 data_dict
        #print(data_dict)
        return True, data_dict  # Return the dictionary and a flag indicating no error
    
    except json.JSONDecodeError as e:
        if isinstance(input_json, (str, bytes)):
            data_dict = eval(input_json)
            return True, data_dict
        else:
            return False, f"{input_json} || 无法解析"
'''
def parse_json(input_json):
    true, false = True, False
    # print(f"原始输入: {input_json}")  # 打印输入以便调试
    if isinstance(input_json, dict):
        print(f"输入是字典: {input_json}")  # 如果输入是字典，打印
        return True, input_json
    
    input_json = input_json.replace("```json", "").replace("```JSON", "").replace("```", "")
    try:
        # Attempt to parse the JSON input
        
        # print(f"提取的 JSON: {input_json}")  # 打印提取的 JSON 字符串
        data_dict = json.loads(input_json)
        return True, data_dict  # Return the dictionary and a flag indicating no error
    
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")  # 打印解析错误信息
        if isinstance(input_json, (str, bytes)):
            try:
                data_dict = eval(input_json)
                print(f"使用 eval 解析后的数据字典: {data_dict}")  # 打印使用 eval 解析后的结果
                return True, data_dict
            except Exception as eval_error:
                print(f"Eval Error: {eval_error} | {input_json}")  # 打印 eval 错误信息
                return False, f"{input_json} || 无法解析"
        else:
            print(f"返回无法解析：{input_json}")
            return False, f"{input_json} || 无法解析"

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
