$Natural Language Toolkit$

NLTK is a leading platform for building Python programs to work with human language data.

```python
pip show nltk
pip install nltk
```

```python
import nltk
nltk.download()
```
`报错：`
[nltk_data] Error loading punkt: <urlopen error [Errno 11004]
[nltk_data]     getaddrinfo failed>

（1）应该是从 https://raw.githubusercontent.com/nltk/nltk_data/ 下载数据失败，
（2）然后去 https://gitee.com/qwererer2/nltk_data/tree/gh-pages/ 下载整个仓库。_653mb_
（3）将packages下所有文件，复制到jupyter找得到的路径下（文件夹名为nltk_data，可自己创建）。
（4）将`nltk_data\tokenizers` 下的 `punkt.zip` 解压到当前目录即可。
（5）运行测试代码，环境基本好了

```python
import nltk
sentence = "What a happy day"
tokens = nltk.word_tokenize(sentence)
tokens

# Out: ['What', 'a', 'happy', 'day']
```


**参考资料：**
- nltk官网: http://www.nltk.org/


**推荐阅读：**
- [15分钟入门NLP神器—Gensim](https://www.jianshu.com/p/9ac0075cc4c0)
- [一文读懂BERT–原理篇](https://blog.csdn.net/jiaowoshouzi/article/details/89073944)