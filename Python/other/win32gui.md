

win32的python接口， [github地址](https://github.com/mhammond/pywin32)



## 遍历窗口

在 windows 中，窗口句柄 HWND 是一个唯一的标识符，用于引用窗口。


```python
import win32gui

def enum_windows_callback(hwnd, top_windows):
    if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) != "":
        top_windows.append((hwnd, win32gui.GetWindowText(hwnd)))

top_windows = []
win32gui.EnumWindows(enum_windows_callback, top_windows)

for hwnd, title in top_windows:
    print(f"句柄: {hwnd}, 标题: {title}")
```


## 获取窗口位置


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


## 获取窗口、句柄信息

```python
hwnd = win32gui.FindWindow(None, WINDOW_NAME)
gvars.genshin_window_rect = win32gui.GetWindowRect(hwnd)

hwnd = win32gui.FindWindow(None, WINDOW_NAME)
# hwnd = win32gui.GetDesktopWindow()
wDC = win32gui.GetWindowDC(hwnd)
dcObj = win32ui.CreateDCFromHandle(wDC)
cDC = dcObj.CreateCompatibleDC()
dataBitMap = win32ui.CreateBitmap()
```

DC Device Context, 设备上下文是一个用于绘制图形的内存区域。