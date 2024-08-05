
使用 win32api 进行键鼠操作

```python
def mouse_down(x, y, button=MOUSE_LEFT):
    time.sleep(0.1)
    xx,yy=x+gvars.genshin_window_rect[0], y+gvars.genshin_window_rect[1]
    win32api.SetCursorPos((xx,yy))
    win32api.mouse_event(mouse_list_down[button], xx, yy, 0, 0)


def mouse_move(dx, dy):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)

def mouse_up(x, y, button=MOUSE_LEFT):
    time.sleep(0.1)
    xx, yy = x + gvars.genshin_window_rect[0], y + gvars.genshin_window_rect[1]
    win32api.SetCursorPos((xx, yy))
    win32api.mouse_event(mouse_list_up[button], xx, yy, 0, 0)

def mouse_click(x, y, button=MOUSE_LEFT):
    mouse_down(x, y, button)
    mouse_up(x, y, button)
```

