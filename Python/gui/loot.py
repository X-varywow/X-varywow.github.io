# run with admin privileges

import customtkinter
import pyautogui
import threading
import keyboard
from loguru import logger
import time
import random


"""
tmp: utils
"""
# 0.1 -> +-0.1
def get_random(ratio):
    return (random.random()-0.5)*ratio*2



"""
part1. public constant and class
"""

BTN = customtkinter.CTkButton


global_btn ={

}

def log_to_panel(msg):
    print(msg)

class CommonButton():
    """
    启动一个线程，完成传入的 func, 并提供开关，绑定按键;
    通用的功能注册，都用按键形式完成；

    params; time_sleep; if None, means call once per click;
    """
    def __init__(self, root, info, key, func, time_sleep = None):
        self.button = BTN(root, text="ON", command=self.ToggleButton)
        self.button.place(relx=0.5, rely=0.5, anchor=customtkinter.CENTER)

        self.run = False
        self.func = func
        self.info = info
        self.time_sleep = time_sleep if time_sleep else 999999

        threading.Thread(target=self.BtnFunc).start()
        keyboard.add_hotkey(key, self.ToggleButton)

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
            tmp_sleep = self.time_sleep * (1+get_random(0.1))
            pyautogui.sleep(tmp_sleep)



"""
part2. funcs
"""

def auto_pick():
    pyautogui.press('a')


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

root = customtkinter.CTk()
root.geometry("300*150")
root.title("AutoTorchlight")

CommonButton(root, "自动拾取", "f12", auto_pick, time_sleep = 0.15)

root.attributes('-topmost', True)
root.attributes('-alpha', 0.8)
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