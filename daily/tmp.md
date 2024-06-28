
```python
# 获取窗口句柄，定位
# 固定功能，固定实现：状态监控
# ui 显示

# 主1：寻路
# 主2：拾取， OCR 调整


```


## win32gui

```python
import win32gui

def get_window_position(hwnd):
    # 获取窗口的矩形区域
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    return (left, top), (right, bottom)

def find_window(title):
    # 寻找窗口
    hwnd = win32gui.FindWindow(None, title)
    if hwnd:
        return hwnd
    else:
        return None

# 替换下面的'窗口标题'为你想要查找的窗口的标题
title = '窗口标题'
hwnd = find_window(title)

if hwnd:
    position = get_window_position(hwnd)
    print(f"窗口位置: {position}")
else:
    print("没有找到窗口")
```


遍历窗口
```python
def enum_windows_callback(hwnd, top_windows):
    if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) != "":
        top_windows.append((hwnd, win32gui.GetWindowText(hwnd)))

top_windows = []
win32gui.EnumWindows(enum_windows_callback, top_windows)

for hwnd, title in top_windows:
    print(f"句柄: {hwnd}, 标题: {title}")
```

```python
>>> import psutil
>>> list(psutil.win_service_iter())
[<WindowsService(name='AeLookupSvc', display_name='Application Experience') at 38850096>,
 <WindowsService(name='ALG', display_name='Application Layer Gateway Service') at 38850128>,
 <WindowsService(name='APNMCP', display_name='Ask Update Service') at 38850160>,
 <WindowsService(name='AppIDSvc', display_name='Application Identity') at 38850192>,
 ...]
>>> s = psutil.win_service_get('alg')
>>> s.as_dict()
{'binpath': 'C:\\Windows\\System32\\alg.exe',
 'description': 'Provides support for 3rd party protocol plug-ins for Internet Connection Sharing',
 'display_name': 'Application Layer Gateway Service',
 'name': 'alg',
 'pid': None,
 'start_type': 'manual',
 'status': 'stopped',
 'username': 'NT AUTHORITY\\LocalService'}
```



## opencv

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg', 0)
template = cv2.imread('face.jpg', 0)
h, w = template.shape[:2]  # rows->h, cols->w

# 相关系数匹配方法：cv2.TM_CCOEFF
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

left_top = max_loc  # 左上角
right_bottom = (left_top[0] + w, left_top[1] + h)  # 右下角
cv2.rectangle(img, left_top, right_bottom, 255, 2)  # 画出矩形位置
```

## ahk

ahk

pyautogui

## 内存

