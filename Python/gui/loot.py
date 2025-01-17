# run with admin privileges

import customtkinter
import pyautogui
import threading
import keyboard
from loguru import logger
import time
import random
from collections import defaultdict
from PIL import ImageGrab
import pytesseract


"""
tmp: utils
"""
# 0.1 -> +-0.1
def get_random(ratio):
    return (random.random()-0.5)*ratio*2

def random_sleep(t, roll = 0.1):
    pyautogui.sleep(t*(1+get_random(roll)))


"""
part1. public constant and class
"""

BTN = customtkinter.CTkButton


global_btn ={

}

class LogPanel:
    def __init__(self, root):
        self.text = customtkinter.CTkTextbox(root, height=10, state="disabled")
        self.text.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

    def log(self, msg):
        self.text.configure(state="normal")
        self.text.insert("end", f"{msg}\n")  # 插入日志
        self.text.yview("end")  # 滚动到最底部
        self.text.configure(state="disabled")


# 日志功能实现
log_panel = None


def log_to_panel(msg):
    if log_panel:
        log_panel.log(msg)
    print(msg)  # 控制台日志作为备用


class ItemPanel:
    def __init__(self, root):
        self.text = customtkinter.CTkTextbox(root, height=10, state="disabled")
        self.text.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

    def update_items(self, item_dict):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")  # 清空当前内容
        for item, count in item_dict.items():
            self.text.insert("end", f"{item}: {count}\n")  # 显示物品及数量
        self.text.configure(state="disabled")


class CommonButton:
    """
    启动一个线程，完成传入的 func, 并提供开关，绑定按键;
    通用的功能注册，都用按键形式完成；

    params; time_sleep; if None, means call once per click;
    """

    def __init__(self, root, info, key, func, time_sleep=None):
        self.button = BTN(root, text=f"{info} ON", command=self.ToggleButton)
        self.button.grid(row=0, column=len(global_btn), padx=10, pady=10, sticky="w")
        global_btn[key] = self.button

        self.run = False
        self.func = func
        self.info = info
        self.time_sleep = time_sleep if time_sleep else 999999

        threading.Thread(target=self.BtnFunc, daemon=True).start()
        keyboard.add_hotkey(key, self.ToggleButton)  # 绑定快捷键

    def ToggleButton(self):
        if self.run:
            self.run = False
            self.button.configure(text=f"{self.info} ON")
            log_to_panel(f"{self.info} 已关闭")
        else:
            self.run = True
            self.button.configure(text=f"{self.info} STOP")
            log_to_panel(f"{self.info} 已打开")

    def BtnFunc(self):
        while self.run:
            self.func()
            random_sleep(self.time_sleep)


"""
part2. funcs
"""


def auto_pick():
    pyautogui.press("a")

# 物品统计数据
item_stats = defaultdict(int)

def ocr_scan():
    """
    OCR 扫描屏幕右下角区域，并统计识别到的物品。
    """
    # 设置扫描区域 (右下角)
    screen_width, screen_height = pyautogui.size()
    scan_region = (screen_width - 300, screen_height - 200, screen_width, screen_height)

    # 截图并进行 OCR 识别
    screenshot = ImageGrab.grab(bbox=scan_region)
    text = pytesseract.image_to_string(screenshot, lang="eng")

    # 更新物品统计
    for line in text.splitlines():
        if line.strip():  # 忽略空行
            item_stats[line.strip()] += 1

    # 更新物品显示
    if item_panel:
        item_panel.update_items(item_stats)

    log_to_panel("OCR 扫描完成并更新物品信息")

def auto_trade(amount_limit, cost_limit, page_limit = 30):
    res = []
    page_cnt = 0

    def check_end():
        ocr_end()

    while check_end() and page_cnt < page_limit:
        page_cnt += 1
        data = ocr_window()
        res.append(data)

    res.sort()
    while check_begin():
        buy(res[0], amount_limit, cost_limit)


def auto_wanjie():
    pyautogui.press('d')
    pyautogui.click(x=500, y=600)
    pyautogui.click(x=500, y=600)
    time.sleep(20)

    while check_finish():
        time.sleep(1)
    
    for i in range(5):
        pyautogui.press('a')
        time.sleep(0.2)
    status, pos = ocr_window()
    pyautogui.press('d')


def auto_stat():
    pass


"""
part3. windows
"""

# 创建窗口
root = customtkinter.CTk()
root.geometry("400x300")
root.title("AutoTorchlight")

# 设置日志面板
log_panel = LogPanel(root)
item_panel = ItemPanel(root)

# 添加功能按钮
CommonButton(root, "自动拾取", "f12", auto_pick, time_sleep=0.15)
CommonButton(root, "实时扫描", "f11", ocr_scan, time_sleep=2) 

# 调整布局
root.grid_rowconfigure(1, weight=1)  # 日志区域可扩展
root.grid_rowconfigure(2, weight=1)  # 物品区域可扩展
root.grid_columnconfigure(0, weight=1)

# 窗口属性
root.attributes("-topmost", True)
root.attributes("-alpha", 0.8)
root.mainloop()



# class AutoTorchlight:
#     def __init__(self, root):
#         self.root = root
#         root.geometry("300*150")
#         root.title("AutoTorchlight")

#         CommonButton(root, "自动拾取", "f12", lambda: pyautogui.press('a'))


#         self.root.attributes('-topmost', True)
#         self.root.attributes('-alpha', 0.8)

# if __name__ == "__main__":
#     root = customtkinter.CTk()
#     app = AutoTorchlight(root)
#     root.mainloop()