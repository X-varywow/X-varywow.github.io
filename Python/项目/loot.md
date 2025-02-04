
可实现功能：
- 自动拾取
- 拍卖行自动低价




```python
import tkinter as tk
import pyautogui
import threading
import keyboard

class AutoClickerA:
    def __init__(self, root):
        self.root = root
        root.geometry("450x450")
        root.title("自动按A拾取")
        self.run = False
        self.button = tk.Button(root, text="ON", command=self.ToggleButton)
        self.button.place(x=100, y=100, width=250, height=250)
        keyboard.add_hotkey('f12', self.ToggleButton)

    def ToggleButton(self):
        if not self.run:
            self.run = True
            self.button.config(text="STOP")
            threading.Thread(target=self.AutoClick).start()
        else:
            self.run = False
            self.button.config(text="ON")

    def AutoClick(self):
        while self.run:
            pyautogui.press('a')
            pyautogui.sleep(0.1)

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoClickerA(root)
    root.mainloop()
```



不生效常见解决方案：
- 管理员身份运行