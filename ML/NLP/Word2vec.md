
## 1. 介绍1

`Word2Vec` 是语言模型中的一种，是将词转化为「可计算」「结构化」的向量的过程。

这种方式在 2018 年之前比较主流，但是随着 BERT、GPT2.0 的出现，这种方式已经不算效果最好的方法了。

**训练模式：**
- `CBOW` (Continuous Bag-of-Words Model)
  - 通过上下文来预测当前值。
- `Skip-gram` (Continuous Skip-gram Model)
  - 用当前词来预测上下文

**优点：**
- 由于 Word2vec 会考虑上下文，跟之前的 Embedding 方法相比，效果要更好（但不如 18 年之后的方法）
- 比之前的 Embedding 方法维度更少，所以速度更快
- 通用性很强，可以用在各种 NLP 任务中

**缺点：**
- 由于词和向量是一对一的关系，所以多义词的问题无法解决。
- Word2vec 是一种静态的方式，虽然通用性强，但是无法针对特定任务做动态优化


**优化方法：**
- 负例采样
- 层序Softmax

参考资料：
- [easyai-Word2vec](https://easyai.tech/ai-definition/word2vec/)

## 2. 介绍2

就像一把剑，总可以用（攻击力、耐久度、重量）将其表示出来。同样，一个单词，也可以使用向量来表示。

有了这种基本思想，不难理解：

$$vec(“king”) - vec(“man”) + vec(“woman”) \approx vec(“queen”)$$


但实际的处理过程，仍然是一团浆糊，对吧。

------------------

我们大概知道，这其中涉及到了一个事物：文本表示（将非结构化的数据转化为结构化的数据）

几种基本方法：独热编码、整数编码，词嵌入；具体参考：[easyai-词嵌入Word embedding](https://easyai.tech/ai-definition/word-embedding/)

<br>

_词袋模型 | Bag-of-words_

```python
# input: John likes to watch movies. Mary likes movies too.
#        John also likes to watch football games. Mary hates football.

# output the vectors:
[1, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0]
[1, 1, 1, 1, 0, 1, 0, 1, 2, 1, 1]
```

在示例中，词袋序列是：`["John", "likes", "to", "watch", "movies", "Mary", "too", "also", "football", "games", "hates"]`

该模型的缺点：（丢失了句子中单词顺序的信息）（不学习单词的含义）

好了，我们现在大概知道了文本表示这么一回事，但离目标还是很远。



## 3. 任务目标

- 推导并实现 skip-gram
  - 写出 skip-gram 未进行负采样（原始、课上讲）和进行负采样后详细的推导过程；
  - 分别实现未进行负采样和进行了负采样之后的 skip-gram 模型。
- 用余弦相似度给出最相似的单词
  - （dog, whale, before, however, fabricate）
- 用向量加减给出结果

> 第一个问题：（找 gensim 源代码）or（论文复现）or（网上找代码）<br> 最后两个问题是教材中的，找不到的

## 4. 介绍3

- https://jalammar.github.io/illustrated-word2vec/
- [词向量word2vec模型解读](https://www.bilibili.com/video/BV1F7411x749)⭐
- [word2vec训练过程及代码](https://www.bilibili.com/video/BV11U4y1E7sK)⭐
- [神经网络相关](https://www.bilibili.com/video/BV1SJ411i7Aq)
- [word2vec 中的数学原理详解](https://www.cnblogs.com/peghoty/p/3857839.html)
- [Word2Vec-知其然知其所以然](https://www.zybuluo.com/Dounm/note/591752)



## 5. 代码参考

```python
# 照着第二个星号的视频写的，没有优化：负采样，层序softmax，效率极低
import numpy as np
from collections import defaultdict, Counter
import pickle
from tqdm import tqdm

def softmax(x):
    ex = np.exp(x)
    return ex/np.sum(ex,keepdims = True)

def get_cos_similar(v1, v2):
    num = float(np.dot(v1, v2))
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


class word2vec():
    def __init__(self, rows = 1, window_size = 5, embedding_num = 50, with_neg = True, neg_rate = 5, min_cnt = 0):
        self.rows = rows
        self.window_size = window_size  # 窗口大小
        self.embedding_num = embedding_num # 词向量维度
        self.with_neg = with_neg # 是否负采样
        self.neg_rate = neg_rate # 负样本比率
        self.min_cnt = min_cnt # 最小词频阈值
        self.vocab = Counter()
        
        self.index2word = []
        self.word2index = {}
        self.word2onehot = {}
        
        # 是（input,ouput）的正样本
        self.samples = defaultdict(list)
        
        # 负样本
        self.neg_samples = defaultdict(list)
        
        # 词向量矩阵 embedding
        # 每一行是一个单词的 词向量
        self.w1 = []
        
        # context
        self.w2 = []
    
    def process_data(self, data):
        d = Counter(data)
        return [word for word in data if d[word] > self.min_cnt]
              
    def get_data(self, path):
        vis = set()
        cnt = 0
        with open(path, encoding="UTF-8") as f:
            for i in range(self.rows):
                data = f.readline().split()
                #print(data)
                for i,word in enumerate(data):
                    #self.vocab[word] += 1
                    
                    # 构造训练数据
                    for j in range(max(0,i-self.window_size//2),min(len(data),i+self.window_size//2+1)):
                        if i != j:
                            self.samples[data[i]].append(data[j])
                    

                    # 构造字典
                    if word not in vis:
                        vis.add(word)
                        self.index2word.append(word)
                        self.word2index[word] = cnt
                        cnt += 1
                        
        self.vocab_size = cnt
        for i,word in enumerate(self.index2word):
            self.word2onehot[word] = np.array([[int(idx==i) for idx in range(self.vocab_size)]])
        #print(self.index2word, self.word2index, self.vocab_size)
        #print(self.word2onehot)
        #print(self.samples)
        
        # 根据 vocab_size， embedding_num 初始化两个矩阵
        self.w1 = np.random.normal(-1,1,size = (self.vocab_size, self.embedding_num))
        self.w2 = np.random.normal(-1,1,size = (self.embedding_num, self.vocab_size))

    def train(self, path):
        self.get_data(path)
        self.vocab_size
        self.embedding_num
        self.lr = 0.01
        self.epoch = 1
        
        for e in range(self.epoch):
            print("epoch:{}".format(e))
            for in_word in tqdm(self.samples):
                for out_word in self.samples[in_word]:
                    in_onehot, out_onehot = self.word2onehot[in_word], self.word2onehot[out_word]

                    hidden = in_onehot @ self.w1
                    p = hidden @ self.w2
                    prediction = softmax(p)

                    #print(hidden.shape, self.w2.shape, p.shape, prediction.shape)

                    #print(hidden,prediction)
                    #反向传播，即根据 预测值和实际值，更新模型参数
                    # 涉及到矩阵求导，
                    # A @ B = C
                    # delta_C = G
                    # delta_A = G @ B.T
                    # delta_B = A.T @ G

                    G2 = prediction - out_onehot
                    delta_w2 = hidden.T @ G2
                    G1 = G2 @ self.w2.T
                    #print(in_onehot.shape, G1.shape)
                    delta_w1 = in_onehot.T @ G1

                    self.w1 -= self.lr * delta_w1
                    self.w2 -= self.lr * delta_w2
                    
                
    def save(self, file):
        with open(file, "w",encoding='utf-8') as f:
            f.write("{} {}\n".format(self.vocab_size, self.embedding_num))
            for i, word in enumerate(self.index2word):
                f.write(word + " " +" ".join([str(num) for num in self.w1[i]]))
                f.write("\n")
                
        
    def get_vec(self, idx):
        return self.w1[idx]
                
    # 获取前 n 个最相似的词
    def sim(self, word, n = 10):
        idx = self.word2index[word]
        res = []
        for i in range(self.vocab_size):
            if i == idx:
                continue
            res.append([self.index2word[i],get_cos_similar(self.get_vec(i),self.get_vec(idx))])
        
        print(sorted(res, key = lambda x: x[1], reverse = True)[:n])
        

if __name__ == "__main__":
    path = "word2vec_data\训练语料.txt"
    model = word2vec(rows=1)
    model.train(path)
    model.save("word2vec.txt")
```


?>**训练过程的优化**：
<br><br>**层序softmax** 将 最后的 softmax 多分类变成二分类，涉及到哈夫曼树；使 softmax 复杂度从 `n` 降低为 `logn`，不使用 softmax 函数而使用 sigmoid。
<br><br>**负采样**，可以使样本标签不总为 1.

```python
# 读模型并测相似，无优化
import numpy as np

def get_cos_similar(v1, v2):
    num = float(np.dot(v1, v2))
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

d = {}

def read_vec(path):
    with open(path, encoding="UTF-8") as f:
        headline = f.readline()
        while data := f.readline():
            data = data.split()
            if data:
                d[data[0]] = np.array([float(num) for num in data[1:]])
        
def most_similar(word, topn = 10):
    res = []
    for cur in d:
        if cur == word:
            continue
        else:
            res.append([cur, get_cos_similar(d[cur], d[word])])
    res.sort(key = lambda x:x[1], reverse = True)
    return res[:topn]

if __name__ == "__main__":
    read_vec("word2vec.txt")
    print(most_similar("dog"))
```

## 6. 使用 gensim

```python
from gensim.models import word2vec

class MySentence:
    def __init__(self, data_path, max_line=None):
        self.data_path = data_path
        self.max_line = max_line
        self.cur_line = 0

    def __iter__(self):
        if self.max_line is not None:
            for line in open(self.data_path, 'r', encoding='utf-8'):
                if self.cur_line >= self.max_line:
                    return
                self.cur_line += 1
                yield line.strip('\n').split()
        else:
            for line in open(self.data_path, 'r', encoding='utf-8'):
                yield line.strip('\n').split()

ms = MySentence("word2vec_data\训练语料.txt")
model = word2vec.Word2Vec(ms, min_count=1)
```

```python
# 保存模型（词向量）
model.wv.save_word2vec_format("vec.txt")

# 词向量应用：查看相似的单词
model.wv.most_similar("dog")

model.wv.most_similar(positive=['French', 'England'], negative=['wine'],topn=3)
```


参考资料：
- https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html

参考论文：
- [Efficient Estimation of Word Representations in Vector Space](http://cn.arxiv.org/pdf/1301.3781v3.pdf)
- [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)
