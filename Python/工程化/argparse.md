

[官方文档](https://docs.python.org/zh-cn/3/library/argparse.html)

[教程1](https://www.liaoxuefeng.com/wiki/1016959663602400/1529653965619235)

`argparse` 用于命令行程序中，获取命令行参数。


```python
import argparse

# STEP1: 创建一个解析器
parser = argparse.ArgumentParser(
    prog = "your_program_name",
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
)


# STEP2: 添加参数
parser.add_argument("-e", "--enc_model_fpath", type=Path,
                    default="saved_models/default/encoder.pt",
                    help="Path to a saved encoder")

parser.add_argument("-s", "--syn_model_fpath", type=Path,
                    default="saved_models/default/synthesizer.pt",
                    help="Path to a saved synthesizer")

parser.add_argument("-v", "--voc_model_fpath", type=Path,
                    default="saved_models/default/vocoder.pt",
                    help="Path to a saved vocoder")

parser.add_argument("--cpu", action="store_true", help=\
    "If True, processing is done on CPU, even when a GPU is available.")

parser.add_argument("--no_sound", action="store_true", help=\
    "If True, audio won't be played.")

parser.add_argument("--seed", type=int, default=None, help=\
    "Optional random number seed value to make toolbox deterministic.")


# STEP3: 解析参数
args = parser.parse_args()

# vars() 返回对象object的属性和属性值的字典对象。
arg_dict = vars(args)

print(args.enc_model_fpath)


# Hide GPUs from Pytorch to force CPU processing
if arg_dict.pop("cpu"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

ArgumentDefaultsHelpFormatter 自动添加默认的值的信息到每一个帮助信息的参数中


```bash
# 参数正确，正确解析；只有当参数全部有效时，才会返回一个NameSpace对象。
main.py -e dir1

# 参数错误，打印错误信息
main.py -e

# 使用 -h, 打印帮助信息
main.py -h
```



-------


```python
# sys.argv 也可获取命令行参数

import sys
print(sys.argv)
```
