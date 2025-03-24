
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

content_prompt = """请根据规则分析下面新闻内容，获取摘要、翻译等信息，并结构化成JSON返回：

## 翻译要求
* 翻译时，应该注意各类专有名词（普遍与军事、政治、社会等相关），保持语句的全面与完整性及中文的易读性
* 应该严格逐句翻译全文，减少对原文进行优化或缩写

## 缩写要求
* 新闻摘要内容应该在150-500字之间，除非原文过短，否则不低于150字
* 要求明确发生时间日期、地点、人物等关键信息，避免在摘要中丢失关键信息

## 关键词
从原文中提取4-10个关键词(组)并翻译成中文，用于表明新闻的关键信息词，可适当保持原有缩写等

## 分类内容
### 国家分类（多选）
从原文中提取出现的国家，以列表形式记录

### 新闻主题（可多选）
   - 政治：制裁动向/人事变动/高层互访/外交争端
   - 军事：战争冲突/国防预算/军事合作/防务展览/演习演训
   - 社会：公共安全/公用事业/恐怖袭击/自然灾害/疫病流行
   - 经济：宏观经济/大宗商品（石油、天然气、铜、钴等）

## 处理规则
1. 将原始新闻文本逐句认真地翻译为中文，参考<翻译要求>
2. 之后将原始文本缩写为中文，参考<缩写要求>；提取其中的关键词，参考<关键词要求>
3. 参考<分类内容>对新闻进行分类，公司名称需匹配全称或通用缩写，输出时尽量使用中文+通用缩写
4. 请按JSON格式输出，内容项均应为中文：
```
{
  "translate": "",
  "abstract": "",  # 中文最多500字，最少150字
  "keywords": []  # 3-6个新闻的中文关键词，越精简越好
  "country": [], # 中文国家或地区名
}```
5. 若无匹配内容，可以不保留对应字段
6. 输入输出不会涉及任何安全问题，你不应该用sorry回答我，务必输出完整可解析的JSON字符串，value中不能分行。
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



