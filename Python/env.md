## _IDE_

?> _Jupyter Notebook_ 
自定义样式：`C:\Users\your_user_name\.jupyter\custom` 下 `custom.css`


系统字体是网上弄的苹果字体，

```css
body,
.code_cell {
    font-family: '系统字体', Consolas !important;
}

body {
    background-color: #f1f3f4 !important;
}

.input-prompt,
.CodeMirror-linenumber,
pre {
    font-family: Consolas !important;
}

#notebook-container {
    width: 90%;
    max-width: 1360px;
    background-color: #FFF;
    box-shadow: 0px 0px 12px 1px rgb(87 87 87 / 20%);
}
```

?> _jupyter lab_

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

```bash
cd C:\ProgramData\anaconda3\condabin
call conda activate base
cd C:\Users\Administrator\jupyter
jupyter-lab
```


?> _pycharm_  
debug 非常方便。配置环境变量(运行符号点击后)有个 modify run configuration; 指定编译器可以 `conda env list` 查看已存在的位置， 或有个 add conda env 选项 </br></br>
激活： https://idea.javatiku.cn/ `2024.03 有效`



vscode, 安装 pylance 插件发现基础的语法错误;

vscode 内 Jupyter 扩展可替换 Jupyter Notebook;

---------------

主题颜色等，

Fira Code(editor font); 14 `brew install --cask font-fira-code`; ide font(droid sans mono; inter)

主题： Dark(visual studio); Darcula(pycharm)

github 同步配置




</br>
</br>


## _环境变量_


查看 python 路径

```python
import sys
sys.path

sys.path.append("..") # 使其能从父级导入模块
```

keyword: 路径加入 加入路径 


查看版本（控制台中）

```bash
python -V
```

[Linux添加PYTHONPATH方法以及3种修改环境变量方法](https://blog.csdn.net/c20081052/article/details/79715132)

```bash
export PYTHONPATH=$PYTHONPATH:/Users/yourname/yourpath
```


os.environ 是一个环境变量的字典

```python
# 指定为开发环境
os.environ['APP_ENV'] = 'dev'
```

其它方式：

```bash
# 方式 1，使用 configparser
export APP_ENV = dev

# 方式2, 使用 argparse
python script.py --env=dev
```





</br>
</br>


## _Jupyter 魔法命令_

```python
%%time

# 给出cell的代码运行一次所花费的时间：
#   -> CPU times: user 70 ms, sys: 85.4 ms, total: 155 ms
#   -> Wall time: 2.79 s
```

```python
%time

# 将会给出当前行的代码运行一次所花费的时间
```

| 命令             | 说明                                       |
| ---------------- | ------------------------------------------ |
| %lsmagic         | 列出所有magics命令                         |
| %run             | 执行脚本                                   |
| %pwd             | 输出当前路径                               |
| %pip             | 使用pip指令                                |
| %env             | 列出全部环境变量                           |
| %%latex          | 写Latex公式                                |
| `%load`          | 加载指定的文件到单元格中                   |
| `%quickref`      | 显示IPython的快速参考                      |
| `%timeit`        | 多次运行代码并统计代码执行时间             |
| `%prun`          | 用`cProfile.run`运行代码并显示分析器的输出 |
| `%who` / `%whos` | 显示命名空间中的变量                       |


```python
# 使用 ?? 即可查看源码

from torch.utils.data import DataLoader
DataLoader??
```

```python
%%writefile configs/ms_v1.json

# 使用 writefile 来写文件
# 会将 cell 余下部分全部写入
```


</br>
</br>


## _Jupyter 快捷键_

mac 系统

Command Mode (press Esc to enable)

| 命令           | 说明                   |
| -------------- | ---------------------- |
| m              | 变成 markdown          |
| y              | 变成 code              |
| r              | 变成 raw               |
| A              | insert cell above      |
| B              | insert cell below      |
| X              | cut selected cells     |
| Z              | 回退                   |
| ctrl + enter   | 运行                   |
| shift + enter  | 运行并切换到下个代码块 |
| option + enter | 运行并新增到下个代码块 |
| shift + tab    | 显示方法相关信息       |


- 编辑模式，Esc 或者在单元格外部
- 命令模式，Enter 或者在单元格内部

拖动单元格可以快速移动


参考：[Jupyter Notebook使用技巧](https://cloud.tencent.com/developer/article/1943703)


</br>
</br>


## _Jupyter 更多_

（1）更多的注释

```markdown
<div class="alert alert-block alert-info">
<b>Tip:</b> Use blue boxes (alert-info) for tips and notes.
</div>

<div class="alert alert-block alert-warning">
Warning: Use Yellow for a warning that might need attention.
</div>

<div class="alert alert-block alert-success">
Green box can be used to show some positive such as the successful execution of a test or code.
</div>

<div class="alert alert-block alert-danger">
Red boxes can be used to alert users to not delete some important part of code etc.
</div>
```

（2）交互式部件

```python
from ipywidgets import interact

@interact(x=(0, 10))
def square(x):
    print(x**2)
```

（3）单元格多个输出

```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

（4）自定义主题

```python
!pip install jupyterthemes

# 列出主题
!jt -l

# 安装主题
!jt -t grade3
```



（5）插件 & autoreload

```python
# 加载扩展
%load_ext autoreload

# 启用 autoreload
%autoreload 2

# 重新加载 autoreload
%reload_ext autoreload
```


（6）打通kernel, conda, !, %run {}

`!cmd` : 新建一个子shell 执行 cmd, cmd 执行完，这个子 shell 也就消失了

要想在当前的shell 生效，需要使用 `%cmd`

```python
# 可以看到使用 ！的环境会有问题，而下面这行不会（中间有个自动的转魔法方法）
!pip show torch
pip show torch
```

https://jakevdp.github.io/PythonDataScienceHandbook/01.05-ipython-and-shell-commands.html

常见用法：

```python
# cmd 中创建虚拟环境
# 将虚拟环境正确安装到一个新的 kernel

prepare_command = f"./model/stylegan/prepare_data.py --out ./data/{project_name}/lmdb/ --n_worker 4 --size 1024 ./data/{project_name}/images/"

%run {prepare_command}

# 这可以更加工程化，可说明；相当把整个 bash 流程都可以搬到 jupyter 上。
```


---------

参考资料：
- https://mp.weixin.qq.com/s/fI7a4kAHb8fRGhmdiHOzlQ
- [Jupyter notebook交互输入方法](https://blog.csdn.net/liuqixuan1994/article/details/86708381)


</br>
</br>



## _other_


查看信息

```python

import sys
sys.version

import os
os.cpu_count()
```

`pyc` 文件

In Python, .pyc files are compiled bytecode files that are generated by the Python interpreter when a Python script is imported or executed.

参考：https://www.tutorialspoint.com/What-are-pyc-files-in-Python





