from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """## 流程控制

### 条件语句

**条件语句作用：** 条件成立执行某些代码，条件不成立则不执行这些代码。

**1. if语法：**
```python
if 条件:
    条件成立执行的代码1
    条件成立执行的代码2
```

**示例：**
```python
age = int(input('请输入您的年龄：'))
if age >= 18:
    print(f'您的年龄是{age},已经成年，可以上网')
```

**2. if...else...语法：**
```python
if 条件:
    条件成立执行的代码
else:
    条件不成立执行的代码
```

**示例：**
```python
age = int(input('请输入您的年龄：'))
if age >= 18:
    print(f'您的年龄是{age},已经成年，可以上网')
else:
    print(f'您的年龄是{age},未成年，请自行回家写作业')
```

**3. 多重判断：**
```python
if 条件1:
    条件1成立执行的代码
elif 条件2:
    条件2成立执行的代码
else:
    以上条件都不成立执行的代码
```

**示例：**
```python
age = int(input('请输入您的年龄：'))
if age < 18:
    print(f'您的年龄是{age},童工一枚')
elif 18 <= age <= 60:
    print(f'您的年龄是{age},合法工龄')
else:
    print(f'您的年龄是{age},可以退休')
```

**4. if嵌套：**
```python
if 条件1:
    条件1成立执行的代码
    if 条件2:
        条件2成立执行的代码
```

**5. 三目运算符：**
```python
值1 if 条件 else 值2

# 示例
a = 1
b = 2
c = a if a > b else b
```"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,     # 每个块最大字符数
    chunk_overlap=50,   # 相邻块的重叠字符数
    separators=["\n\n", "\n", "。", "！", "？", "；", "```", ""]
)

chunks = splitter.split_text(text)
for i, c in enumerate(chunks):
    print(f"---块{i}({len(c)}字)---")
    print(c)