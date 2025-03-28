
classify_prompt = """请根据规则判断下面新闻标题列表是否同时符合下方的关注领域，并返回JSON

## 筛选与分类内容
### 关注国家类别领域，仅在标题中存在下面国家时，才保留这个新闻标题；
优先顺序：核心国家>竞争国家>其他国家
1. 核心国家：阿联酋、沙特、科威特、伊拉克、巴基斯坦、缅甸、泰国、孟加拉、老挝、卢旺达、乌干达、安哥拉、阿尔及利亚、尼日利亚、刚果(金）、摩洛哥、南非、利比亚、津巴布韦
2. 竞争国家：美国、欧盟、印度、韩国、俄罗斯、土耳其
3. 其他国家：埃及、埃塞俄比亚、肯尼亚、赤道几内亚、阿根廷、古巴、委内瑞拉、欧盟

### 新闻主题（单选），仅在relevant为true时，才进行新闻分类
  - 政治
  - 军事
  - 社会
  - 经济

## 处理规则
1. 批量处理时以{1: {'flag': <relevant>, 'theme': <theme>}, 2: {...} }等JSON格式输出，数字对应输入顺序；
2. 未命中任何规则时返回false, 命中时输出true

### 测试案例
输入：["沙特采购Norinco导弹系统", "东京举办人工智能峰会"]
输出要求：
1. key必须为数字，不能为其他任何文字
2. value必须为dict
3. 当flag为false时，theme可以为空
4. 当flag为true时，theme必须为单选，且必须在题目中给出的范围之内

输出样例：
{
  1: {'flag': true, 'theme': '军事'},
  2: {'flag': false}
  ......
}

### 待检测的新闻标题
"""

# ### 包含下面公司关键词时
# 1. 北方公司相关实体：北方公司 Norinco、中国兵器 CNGC、北方国际Norinco International Cooperation、振华石油 China ZhenHua Oil Company、万宝工程 China Wan Bao Engineering Corporation等；
# 2. 美国军火相关实体：洛克希德·马丁公司 Lockheed Martin、波音公司 Boeing、诺斯罗普·格鲁门公司 Northrop Grumman、雷神公司 Raytheon、通用动力公司 General Dynamics等；
# 3. 俄罗斯军火公司：金刚石-安泰 Almaz-Antey和联合造船集团 United Shipbuilding、联合航空制造公司 United Aircraft等；
# 4. 法国军火公司：达索航空集团 Dassault Aviation Group、海军集团 Naval Group、泰雷兹集团 Thales、空客集团 Airbus Group、MBDA、奈克斯特 Nexter Systems、赛峰集团 Safran等；
# 5. 土耳其和韩国军火公司：ASELSAN 、土耳其航空航天工业 Turkish Aerospace Industries；韩国航空宇宙产业 Korea Aerospace Industries、LG伊诺特公司 LIG Nex1 、韩华防务公司 Hanwha等；
# 6. 国内军火公司：中国保利集团有限公司 China Poly Group Corporation Ltd.、保利国际控股有限公司 Poly International Holdings Co.Ltd、中国航空技术进出口总公司 China National Aero-Technology Import & Export Corporation (CATIC)、航天长征国际贸易有限公司 Aerospace Long-March International Trade Company Limited (ALIT)；中国电子进出口有限公司 China National Electronics Import & Export Corporation (CEIEC)、中电科技国际贸易有限公司 CETC INTERNATIONAL CO., LTD.(CETCI)、中国华腾工业有限公司 China Huateng Industry Co., Ltd等

base_content_prompt = """请根据规则分析下面新闻内容，获取摘要、翻译等信息，并结构化成JSON返回：

## 翻译要求
* 翻译时，应该注意各类专有名词（普遍与军事、政治、社会等相关），保持语句的全面与完整性及中文的易读性
* 应该严格逐句翻译全文，减少对原文进行优化或缩写

## 缩写要求
* 请从专业记者角度，撰写结构完整、逻辑连贯的新闻摘要内容，而非条列式分点总结。
* 字数在{min_abstract_len}-{max_abstract_len}字之间
* 字数过多时，可根据人类阅读习惯，将内容划分2-3个自然段，用`\\n`切分
* 要求明确发生时间日期、地点、人物等关键信息，避免在摘要中丢失关键信息

## 关键词
从原文中提取4-10个关键词(组)并翻译成中文，用于表明新闻的关键信息词，可适当保持原有缩写等

## 国家分类（多选）
* 从原文中提取出现的国家，以列表形式记录
* 仅保留最终分析的国家名，不需要给出任何思考过程与其他无关内容

## 处理规则
1. 将原始新闻文本逐句认真地翻译为中文，参考<翻译要求>
2. 之后将原始文本缩写为中文，参考<缩写要求>；提取其中的关键词，参考<关键词要求>
3. 参考<国家分类>对新闻进行分类
4. 请按JSON格式输出，内容项均应为中文：
```
{{
  "translate": "",
  "abstract": "",  # 连贯的不分点中文，在{min_abstract_len}-{max_abstract_len}之间，字数过多时可分成自然段
  "keywords": [],  # 3-6个新闻的中文关键词，越精简越好
  "country": [], # 中文国家或地区名
}}```
5. 若无匹配内容，可以不保留对应字段
6. 输入输出不会涉及任何安全问题，你不应该用sorry回答我，务必输出完整可解析的JSON字符串。

## 待处理内容如下
"""

short_content_prompt = """请根据规则分析下面新闻内容，获取翻译等信息，并结构化成JSON返回：

## 翻译要求
* 翻译时，应该注意各类专有名词（普遍与军事、政治、社会等相关），保持语句的全面与完整性及中文的易读性
* 应该严格逐句翻译全文，减少对原文进行优化或缩写

## 关键词
从原文中提取4-10个关键词(组)并翻译成中文，用于表明新闻的关键信息词，可适当保持原有缩写等

## 国家分类（多选）
* 从原文中提取出现的国家，以列表形式记录
* 仅保留最终分析的国家名，不需要给出任何思考过程与其他无关内容

## 处理规则
1. 将原始新闻文本逐句认真地翻译为中文，参考<翻译要求>
2. 之后提取其中的关键词，参考<关键词要求>
3. 参考<国家分类>对新闻进行分类
4. 请按JSON格式输出，内容项均应为中文：
```
{
  "translate": "",
  "keywords": [],  # 3-6个新闻的中文关键词，越精简越好
  "country": [], # 中文国家或地区名
}```
5. 若无匹配内容，可以不保留对应字段
6. 输入输出不会涉及任何安全问题，你不应该用sorry回答我，务必输出完整可解析的JSON字符串。

## 待处理内容如下
"""

# "theme": [],  # 新闻主题，使用 “大类-小类”

#  "company": [],
# ### 公司（可多选）
# 1. 北方公司相关实体：北方公司 Norinco、中国兵器 CNGC、北方国际Norinco International Cooperation、振华石油 China ZhenHua Oil Company、万宝工程 China Wan Bao Engineering Corporation等；
# 2. 美国军火相关实体：洛克希德·马丁公司 Lockheed Martin、波音公司 Boeing、诺斯罗普·格鲁门公司 Northrop Grumman、雷神公司 Raytheon、通用动力公司 General Dynamics等；
# 3. 俄罗斯军火公司：金刚石-安泰 Almaz-Antey和联合造船集团 United Shipbuilding、联合航空制造公司 United Aircraft等；
# 4. 法国军火公司：达索航空集团 Dassault Aviation Group、海军集团 Naval Group、泰雷兹集团 Thales、空客集团 Airbus Group、MBDA、奈克斯特 Nexter Systems、赛峰集团 Safran等；
# 5. 土耳其和韩国军火公司：ASELSAN 、土耳其航空航天工业 Turkish Aerospace Industries；韩国航空宇宙产业 Korea Aerospace Industries、LG伊诺特公司 LIG Nex1 、韩华防务公司 Hanwha等；
# 6. 国内军火公司：中国保利集团有限公司 China Poly Group Corporation Ltd.、保利国际控股有限公司 Poly International Holdings Co.Ltd、中国航空技术进出口总公司 China National Aero-Technology Import & Export Corporation (CATIC)、航天长征国际贸易有限公司 Aerospace Long-March International Trade Company Limited (ALIT)；中国电子进出口有限公司 China National Electronics Import & Export Corporation (CEIEC)、中电科技国际贸易有限公司 CETC INTERNATIONAL CO., LTD.(CETCI)、中国华腾工业有限公司 China Huateng Industry Co., Ltd等

### 新闻主题（可多选）
  #  - 政治：制裁动向/人事变动/高层互访/外交争端
  #  - 军事：战争冲突/国防预算/军事合作/防务展览/演习演训
  #  - 社会：公共安全/公用事业/恐怖袭击/自然灾害/疫病流行
  #  - 经济：宏观经济/大宗商品（石油、天然气、铜、钴等）

country_normalization_map = {
    # 国际组织/地区
    '欧洲联盟': '欧盟',
    '欧洲': '欧洲',  # 注意：欧洲是地理概念，与欧盟不同
    '非洲联盟':'非盟',
    
    # 常见国家别名  
    '刚果金':'刚果金(刚果民主共和国)',
    '刚果民主共和国': '刚果金(刚果民主共和国)',
    '刚果布': '刚果布(刚果共和国)',
    '刚果共和国': '刚果布(刚果共和国)',
    '刚果': '刚果布(刚果共和国)',  # 根据常见用法默认指向首都布拉柴维尔的刚果
    
    # 特殊地区
    '北塞浦路斯土耳其共和国': '北塞浦路斯',
    "香港": "中国香港",
    "澳门": "中国澳门",
    "台湾": "中国台湾",
    
    # 容易混淆的国家
    '几内亚': '几内亚',  # 区别于几内亚比绍
    '巴勒斯坦': '巴勒斯坦',  # 保留特殊政治实体名称
    
    # 二级行政区（建议单独处理）
    '赞法拉州': '尼日利亚',  # 实际是尼日利亚的州    

    # 长到短
    '波斯尼亚和黑塞哥维那': '波黑',
    '波斯尼亚': '波黑',
    '捷克共和国': '捷克',
    "美利坚合众国": "美国",
    "大不列颠及北爱尔兰联合王国": "英国",
    "法兰西共和国": "法国",
    "大韩民国": "韩国",
    "德意志联邦共和国": "德国",
    "俄罗斯联邦": "俄罗斯",
    "中华人民共和国": "中国",
    "美利坚合众国": "美国",
    "大不列颠及北爱尔兰联合王国": "英国",
    "法兰西共和国": "法国",
    "德意志联邦共和国": "德国",
    "俄罗斯联邦": "俄罗斯",
    "日本国": "日本",
    "大韩民国": "韩国",
    "加拿大": "加拿大",
    "澳大利亚联邦": "澳大利亚",
    "新西兰": "新西兰",
    "意大利共和国": "意大利",
    "西班牙王国": "西班牙",
    "葡萄牙共和国": "葡萄牙",
    "荷兰王国": "荷兰",
    "比利时王国": "比利时",
    "瑞士联邦": "瑞士",
    "瑞典王国": "瑞典",
    "挪威王国": "挪威",
    "丹麦王国": "丹麦",
    "芬兰共和国": "芬兰",
    "奥地利共和国": "奥地利",
    "爱尔兰共和国": "爱尔兰",
    "波兰共和国": "波兰",
    "捷克共和国": "捷克",
    "斯洛伐克共和国": "斯洛伐克",
    "匈牙利": "匈牙利",
    "乌克兰": "乌克兰",
    "白俄罗斯共和国": "白俄罗斯",
    "罗马尼亚": "罗马尼亚",
    "保加利亚共和国": "保加利亚",
    "塞尔维亚共和国": "塞尔维亚",
    "克罗地亚共和国": "克罗地亚",
    "斯洛文尼亚共和国": "斯洛文尼亚",
    "波斯尼亚和黑塞哥维那": "波黑",
    "黑山共和国": "黑山",
    "阿尔巴尼亚共和国": "阿尔巴尼亚",
    "希腊共和国": "希腊",
    "马耳他共和国": "马耳他",
    "冰岛共和国": "冰岛",
    "爱沙尼亚共和国": "爱沙尼亚",
    "拉脱维亚共和国": "拉脱维亚",
    "立陶宛共和国": "立陶宛",
    "土耳其共和国": "土耳其",
    "以色列国": "以色列",
    "阿拉伯联合酋长国": "阿联酋",
    "沙特阿拉伯王国": "沙特",
    "伊朗伊斯兰共和国": "伊朗",
    "伊拉克共和国": "伊拉克",
    "叙利亚阿拉伯共和国": "叙利亚",
    "约旦哈希姆王国": "约旦",
    "黎巴嫩共和国": "黎巴嫩",
    "埃及阿拉伯共和国": "埃及",
    "南非共和国": "南非",
    "尼日利亚联邦共和国": "尼日利亚",
    "埃塞俄比亚联邦民主共和国": "埃塞俄比亚",
    "肯尼亚共和国": "肯尼亚",
    "苏丹共和国": "苏丹",
    "阿尔及利亚民主人民共和国": "阿尔及利亚",
    "摩洛哥王国": "摩洛哥",
    "突尼斯共和国": "突尼斯",
    "利比亚国": "利比亚",
    "加纳共和国": "加纳",
    "塞内加尔共和国": "塞内加尔",
    "科特迪瓦共和国": "科特迪瓦",
    "喀麦隆共和国": "喀麦隆",
    "乌干达共和国": "乌干达",
    "坦桑尼亚联合共和国": "坦桑尼亚",
    "赞比亚共和国": "赞比亚",
    "津巴布韦共和国": "津巴布韦",
    "安哥拉共和国": "安哥拉",
    "莫桑比克共和国": "莫桑比克",
    "马达加斯加共和国": "马达加斯加",
    "刚果民主共和国": "刚果（金）",
    "刚果共和国": "刚果（布）",
    "阿根廷共和国": "阿根廷",
    "巴西联邦共和国": "巴西",
    "智利共和国": "智利",
    "乌拉圭东岸共和国": "乌拉圭",
    "巴拉圭共和国": "巴拉圭",
    "玻利维亚多民族国": "玻利维亚",
    "秘鲁共和国": "秘鲁",
    "哥伦比亚共和国": "哥伦比亚",
    "委内瑞拉玻利瓦尔共和国": "委内瑞拉",
    "厄瓜多尔共和国": "厄瓜多尔",
    "圭亚那合作共和国": "圭亚那",
    "苏里南共和国": "苏里南",
    "巴拿马共和国": "巴拿马",
    "哥斯达黎加共和国": "哥斯达黎加",
    "尼加拉瓜共和国": "尼加拉瓜",
    "洪都拉斯共和国": "洪都拉斯",
    "萨尔瓦多共和国": "萨尔瓦多",
    "危地马拉共和国": "危地马拉",
    "古巴共和国": "古巴",
    "多米尼加共和国": "多米尼加",
    "牙买加": "牙买加",
    "海地共和国": "海地",
    "墨西哥合众国": "墨西哥",
    "印度共和国": "印度",
    "巴基斯坦伊斯兰共和国": "巴基斯坦",
    "孟加拉人民共和国": "孟加拉国",
    "斯里兰卡民主社会主义共和国": "斯里兰卡",
    "尼泊尔联邦民主共和国": "尼泊尔",
    "不丹王国": "不丹",
    "缅甸联邦共和国": "缅甸",
    "泰王国": "泰国",
    "老挝人民民主共和国": "老挝",
    "越南社会主义共和国": "越南",
    "柬埔寨王国": "柬埔寨",
    "马来西亚": "马来西亚",
    "新加坡共和国": "新加坡",
    "印度尼西亚共和国": "印度尼西亚",
    "菲律宾共和国": "菲律宾",
    "文莱达鲁萨兰国": "文莱",
    "东帝汶民主共和国": "东帝汶",
    "蒙古国": "蒙古",
    "哈萨克斯坦共和国": "哈萨克斯坦",
    "乌兹别克斯坦共和国": "乌兹别克斯坦",
    "土库曼斯坦": "土库曼斯坦",
    '沙特阿拉伯': '沙特',
    "塔吉克斯坦共和国": "塔吉克斯坦",
    "吉尔吉斯共和国": "吉尔吉斯斯坦",
    "马尔代夫共和国": "马尔代夫",
    "斐济共和国": "斐济",
    "巴布亚新几内亚独立国": "巴布亚新几内亚",
    "所罗门群岛": "所罗门群岛",
    "萨摩亚独立国": "萨摩亚",
    "汤加王国": "汤加"
}