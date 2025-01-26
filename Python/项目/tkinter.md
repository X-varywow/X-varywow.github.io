
目标：
- torchlight infinite 挂机
- 显示一些额外信息：（检测掉落，收益；后续挂机时，按了什么键位等）
- 利用 ocr, 色块不对时开启技能


ui 够用就行，主要是实现的功能（自动化，统计信息显示）


## tkinter


```python
import tkinter as tk

# 创建主窗口
window = tk.Tk()
window.title("title")
window.geometry("400x300+100+100")

window.resizable(False, False)  # 禁止窗口大小调整
window.overrideredirect(True)  # 隐藏窗口边框和标题栏

# 设置窗口置顶
window.attributes('-topmost', True)
window.attributes('-alpha', 0.5)

# 在窗口中添加一些控件
label = tk.Label(window, text="这是一个置顶窗口")
label.pack(pady=20)

# 运行主循环
window.mainloop()
```

设置背景：

```python
root.configure(bg='blue')  # 设置窗口背景色为蓝色
label = tk.Label(root, text="Hello", bg='red')  # 设置标签背景色为红色
```

左边显示图片，右边实时更新文本：


```python
import tkinter as tk
from tkinter import PhotoImage
import time

def update_frame(time_label, runtime_label, start_time):
    # 获取当前时间戳
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    # 计算程序运行时间
    elapsed_time = time.time() - start_time
    runtime_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    
    # 更新标签文本
    time_label.config(text=f"当前时间: {current_time}")
    runtime_label.config(text=f"运行时间: {runtime_str}")
    
    # 每1000毫秒（1秒）更新一次
    time_label.after(1000, update_frame, time_label, runtime_label, start_time)

def create_gui():
    # 创建主窗口
    root = tk.Tk()
    root.title("图片和时间展示")
    root.geometry("400x300")  # 设置窗口大小

    # 创建主Frame
    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # 左侧图片部分
    left_frame = tk.Frame(main_frame, width=200, height=300)
    left_frame.pack(side="left", fill="both", expand=True)

    # 加载图片
    try:
        image = PhotoImage(file="a.png")  # 替换为你的图片路径
        img_label = tk.Label(left_frame, image=image)
        img_label.image = image  # 防止图片被垃圾回收
        img_label.pack(expand=True)
    except Exception as e:
        print(f"加载图片时出错: {e}")

    # 右侧文本部分
    right_frame = tk.Frame(main_frame, width=200, height=300)
    right_frame.pack(side="right", fill="both", expand=True, padx=20)

    # 创建时间戳标签
    time_label = tk.Label(right_frame, text="", pady=5)
    time_label.pack(expand=True)

    # 创建运行时间标签
    runtime_label = tk.Label(right_frame, text="", pady=5)
    runtime_label.pack(expand=True)

    # 获取程序开始时间
    start_time = time.time()

    # 开始更新时间
    update_frame(time_label, runtime_label, start_time)

    # 进入消息循环
    root.mainloop()

create_gui()
```

在上述基础上加入 OCR 文本信息：

```python
# 1. 创建 OCR 结果标签
ocr_label = tk.Label(right_frame, text="", pady=5)
ocr_label.pack(expand=True)

# 2. update_frame 中传入

# 3. update_frame
try:
    res = ocr()
except:
    res = 0
ocr_label.config(text=f"{res}")
```

变更为文本：

```python
# init
text_frame = tk.Frame(right_frame)
text_frame.pack(fill="both", expand=True)

text_widget = tk.Text(text_frame, wrap="word")
text_widget.pack(side="left", fill="both", expand=True)

scrollbar = tk.Scrollbar(text_frame, command=text_widget.yview)
scrollbar.pack(side="right", fill="y")

text_widget.config(yscrollcommand=scrollbar.set)


# update 
text_widget.delete(1.0, tk.END)  # 清空文本框
text_widget.insert(tk.END, ocr_text.strip())  # 插入新的 OCR 结果
```

## 测测 OCR

mac 上截图只能截到桌面的图，先测试着


```python
from PIL import ImageGrab
screenshot = ImageGrab.grab(bbox=(40, 0, 400, 20))
screenshot

screenshot?? # 显示详细信息
```

pyautogui 也是直接用的这个方法，timeit 测出来耗时 150ms (mac)

> windows 耗时 40ms; 扩展至 2560*1080 也才 48ms 


```python
from paddleocr import PaddleOCR
from PIL import ImageGrab
import numpy as np

screenshot = ImageGrab.grab(bbox=(40, 0, 400, 20))

# ocr = PaddleOCR(lang='en')
ocr = PaddleOCR() # need to run only once to load model into memory

img_path = 'PaddleOCR/doc/imgs_words_en/word_10.png'

img = np.array(screenshot)

result = ocr.ocr(img, det=False, cls=False)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)
```

> 识别的是准的


识别耗时 200ms; (mac, cpu)

**测测 win**


1 s 更新两帧 就不错了；程序剩余 500 - 200 - 150 = 150ms;



ocr 还是 paddle ocr 好些；简单测了下，


[Tesseract](https://github.com/tesseract-ocr/tesseract) 64k star 

速度上慢了1/2, paddleOCR 200ms 的需要 300ms, 准确率也是比不上

```
[[('Chrome文件修改查看历史记录书签个', 0.9954605102539062)]]
识别成：
'Chrome 文 件 怀\n\n'
```






参考：
- https://github.com/yangshun/2048-python/blob/master/puzzle.py
- https://github.com/babalae/better-genshin-impact