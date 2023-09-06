## 1. 字的统计

> defaultdict 是一个字典的子类，为查询提供默认值

```python
from collections import defaultdict

d = defaultdict(int)
with open('test.txt','r') as f:
    for line in f:
        line = line.strip()
        for c in line:
            d[c] += 1
            
for k, v in sorted(d.items(), key = lambda x:x[1], reverse = True):   #降序输出
    print(k,v)
```

## 2. 简易查单词

> 注意文本的编码格式，一般打开txt，编码格式出现在右下角

```python
from collections import defaultdict
import re

d = defaultdict(int)

with open("dict.txt", "r", encoding="ANSI") as f:
    for line in f:
        line = line.strip()
        w, t = line.split("=>")
        d[w] = t
        
# for k, v in d.items():
#     print(k, v)

while True:
    s = input("Word: ")
    if s in ['q','Q']:
        break
    if d.get(s):
        res = re.sub("@","\n",d[s])
        print(res)
```

## 3. 汉字编码

>汉字编码表中存在空缺，所以用 try 来跳过报错

[struct官方文档](https://docs.python.org/zh-cn/3/library/struct.html)

```python
import struct
cnt = 0

for H in range(0xa1,0xf7):
    for L in range(0xa1,0xfe):
        try:
            word = struct.pack("BB",H,L)
            word = word.decode("gbk")
            print(word)
        except:
            cnt += 1
            
print("有{}个错误".format(cnt))
```

## 4. 小作业：汉字字频统计，半角转全角

> 一级常用汉字：3755个，编码为：`0xB0A0` ~ `0xD7F9`
>print 函数有 file 参数


做法1：

```python
import struct
from collections import defaultdict

def fun(char):                         #半角转全角，这代码抄的
    inside_code=ord(char)
    if inside_code<0x0020 or inside_code>0x7e:   #不是半角字符就返回原来的字符
        return char
    if inside_code==0x0020: #除了空格其他的全角半角的公式为:半角 = 全角-0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code)

all_cnt = 0
d = defaultdict(int)

with open("word.txt", "r", encoding="UTF-8") as f:
    for line in f:
        line = line.strip()
        for char in line:
            char = fun(char)
            d[char] += 1
            all_cnt += 1

for H in range(0xb0,0xd8):
    for L in range(0xa0,0xfa):
        word = struct.pack("BB",H,L)
        word = word.decode("gbk")
        print("{}, 字频为：{}".format(word, d[word]/all_cnt))
```

做法2：
```python
import struct
from collections import defaultdict

def ban_to_quan(char):
    Bytes = char.encode("gbk")     #将字符，编码成字节流
    if Bytes[0]&0x80 == 0:
        tmp = struct.pack("BB", 0xa3, Bytes[0]+128)   
        res = tmp.decode("gbk")    #将字节，解码成字符流
    else:
        res = char
        
    return res
        
def convert(corpus, res):
    d = defaultdict(int)
    
    with open(corpus,"r") as i:    
        for line in i:
            line = line.strip()
            for c in line:
                quan = ban_to_quan(c)
                d[quan] += 1
                
    with open(res,"w") as o:   
        for k,v in d.items():
            Bytes = k.encode("gbk")   # 一个汉字变为2个字节，4个十六进制位
            if Bytes[0] < 0xd8:
                print(k, v, file = o)
        
convert("test.txt","hzlist.txt")
```

## 5. 字节流

```python
import struct

s = "我们在上课"
s = s.encode("gbk")

# for c in s:
#     print("{:#x}".format(c))
    
with open("hzlist.txt", "wb") as f:   # wb 表示以二进制格式打开，并从头开始编辑
    for H in range(0xb0,0xd8):
        for L in range(0xa0,0xff):
            word = struct.pack("BB",H,L)
            f.write(word)
            f.write("\n".encode("gbk"))
```

## 6. 汉字点阵显示

>汉字点阵显示，通常需要 16×16 的点阵，2^(16\*16) = (2^8)^32，需要32个字节

```python
import struct

KEYS = [0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01]

res = [] * 16
for i in range(16):
    res.append([] * 16)
    
def LoadLib(path):
    res = []
    with open(path, "rb") as f:
        for i in range(267616//32):
            buffer = f.read(32)
            dots = struct.unpack("32B", buffer)
            res.append(dots)
    return res

def GetZX(arr, HZ):         #返回汉字在字库的 绝对偏移量
    HZ = HZ.encode('gbk')
    return arr[(HZ[0]-0xa1)*94+HZ[1]-0xa1]

def main(lib, hz):
    arr = LoadLib(lib)
    for h in hz:
        zx = GetZX(arr, h)

        for k, row in enumerate(res):
            for j in range(2):
                for i in range(8):
                    asc = zx[k * 2 + j]
                    flag = asc & KEYS[i]
                    row.append(flag)

    for row in res:
        for i in row:
            if i:
                print('*', end=' ')
            else:
                print(' ', end=' ')
        print()

main("hzk.dat", "圆月")  #会在一行显示
```