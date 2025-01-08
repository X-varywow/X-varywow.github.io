# mac: cpu 400%; mem 870mb; avg_frame: 400ms

import tkinter as tk
from tkinter import PhotoImage
import time
from paddleocr import PaddleOCR
from PIL import ImageGrab, ImageTk
import numpy as np

ocr_client = PaddleOCR()

def update_frame(update_content, start_time, last_frame_time):

    # part1. ocr; 350ms
    try:
        # 截取屏幕的特定区域 (x1, y1, x2, y2)
        screenshot = ImageGrab.grab(bbox=(40, 0, 400, 20))
        img = ImageTk.PhotoImage(screenshot)
        update_content['img_label'].config(image=img)
        update_content['img_label'].image = img

        img_array = np.array(screenshot)
        ocr_text = ocr_client.ocr(img_array, det=False, cls=False)
        update_content['text_widget'].delete(1.0, tk.END)  # 清空文本框
        update_content['text_widget'].insert(tk.END, ocr_text)  # 插入新的 OCR 结果
    except Exception as e:
        update_content['text_widget'].delete(1.0, tk.END)
        update_content['text_widget'].insert(tk.END, f"OCR 出错: {e}")


    # 获取当前时间戳
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    # 计算程序运行时间
    elapsed_time = time.time() - start_time
    runtime_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    
    # 更新时间标签
    update_content['time_label'].config(text=f"当前时间: {current_time}")
    update_content['runtime_label'].config(text=f"运行时间: {runtime_str}; 一帧耗时：{(time.time()-last_frame_time)*1000:.0f} ms")


    # 递归更新
    last_frame_time = time.time()
    update_content['time_label'].after(10, update_frame, update_content, start_time, last_frame_time)

def create_gui():
    # 创建主窗口
    root = tk.Tk()
    root.title("Torchlight Helper")
    root.geometry("800x500")  # 设置窗口大小

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
        img_label.image = image
        img_label.pack(expand=True)
    except Exception as e:
        print(f"加载图片时出错: {e}")

    # 右侧文本部分
    right_frame = tk.Frame(main_frame, width=400, height=300)
    right_frame.pack(side="right", fill="both", expand=True, padx=20)

    # 创建时间戳标签
    time_label = tk.Label(right_frame, text="", pady=5)
    time_label.pack()

    # 创建运行时间标签
    runtime_label = tk.Label(right_frame, text="", pady=5)
    runtime_label.pack()

    # 创建带滚动条的文本框
    text_frame = tk.Frame(right_frame)
    text_frame.pack(fill="both", expand=True)

    text_widget = tk.Text(text_frame, wrap="word")
    text_widget.pack(side="left", fill="both", expand=True)

    scrollbar = tk.Scrollbar(text_frame, command=text_widget.yview)
    scrollbar.pack(side="right", fill="y")

    text_widget.config(yscrollcommand=scrollbar.set)


    start_time = time.time()
    update_content = {
        "time_label": time_label,
        "runtime_label": runtime_label,
        "text_widget": text_widget,
        "img_label": img_label
    }
    update_frame(update_content, start_time, start_time)


    root.attributes('-topmost', True)
    root.attributes('-alpha', 0.8)
    root.mainloop()

create_gui()