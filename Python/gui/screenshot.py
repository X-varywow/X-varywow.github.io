# TODO: 测一下速度vs pillow
# ref: https://cnblogs.com/enumx/p/12342186.html
import os
import random
import time
import traceback
import numpy as np
import win32gui
from ctypes import windll
import win32ui
import win32con
from PIL import Image


def has_title_bar(hwnd):
    # 获取窗口样式
    style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)

    # 判断是否有标题栏
    has_title = (style & win32con.WS_CAPTION) == win32con.WS_CAPTION

    return has_title


def screenshot(hwnd, left=0, top=0, right=0, bottom=0, filename=None,is_top=False):
    try:
        if not is_top:

            rect=win32gui.GetClientRect(hwnd)
            width=rect[2]-rect[0]
            height=rect[3]-rect[1]
            #old_time=time.time()

            # 判断窗口是否可见
            if not win32gui.IsWindowVisible(hwnd):
                return False
            # 创建设备描述表
            hwnd_dc = win32gui.GetDC(hwnd)   #GetWindowDC
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()

            # 创建位图对象
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)

            # 将位图对象绑定到设备描述表
            save_dc.SelectObject(save_bitmap)
            result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(),3)   #0  1 或者3  3没有透明通道信息
            if result == 0:
               print("PrintWindow failed")
               return False
            # 将截图保存到位图对象中
            #save_dc.BitBlt((0, 0), (width, height), mfc_dc, (left, top), win32con.SRCCOPY)#win32con.CAPTUREBLT  win32con.SRCCOPY

            # 将位图对象转换为OpenCV图像
            bmp_info = save_bitmap.GetInfo()
            bmp_str = save_bitmap.GetBitmapBits(True)
            img = np.frombuffer(bmp_str, dtype='uint8').reshape((bmp_info['bmHeight'], bmp_info['bmWidth'], 4))
            img=img[top:bottom, left:right]
            if filename is not None:
                # 保存位图对象到文件
                img_pil = Image.fromarray(img[..., [2, 1, 0]])
                img_pil.save(filename, format='JPEG', quality=90)

            # 删除对象，释放资源
            save_dc.DeleteDC()

            win32gui.ReleaseDC(hwnd, hwnd_dc)
            win32gui.DeleteObject(save_bitmap.GetHandle())
            #print(time.time()-old_time)
            return img
        else:
            #old_time = time.time()
            if has_title_bar(hwnd):
                window_rect = win32gui.GetWindowRect(hwnd)
                client_rect = win32gui.GetClientRect(hwnd)

                # 计算非客户区域的尺寸
                title_bar_width = (window_rect[2] - window_rect[0]) - (client_rect[2] - client_rect[0])
                title_bar_height = (window_rect[3] - window_rect[1]) - (client_rect[3] - client_rect[1])


                width = (right - left)
                height = (bottom - top)
                left=left+window_rect[0]+title_bar_width-8
                top=top+window_rect[1]+title_bar_height-8
            else:
                rect = win32gui.GetWindowRect(hwnd)
                width = (right - left)
                height = (bottom - top)
                left = left + rect[0]
                top = top + rect[1]


            # 获取桌面窗口的句柄
            hdesktop = win32gui.GetDesktopWindow()

            # 获取设备上下文
            desktop_dc = win32gui.GetWindowDC(hdesktop)
            img_dc = win32ui.CreateDCFromHandle(desktop_dc)

            # 创建一个兼容DC
            mem_dc = img_dc.CreateCompatibleDC()

            # 创建一个位图对象
            screenshot = win32ui.CreateBitmap()
            screenshot.CreateCompatibleBitmap(img_dc, width, height)

            # 将位图选入内存DC
            mem_dc.SelectObject(screenshot)

            # 使用BitBlt函数从屏幕复制到内存DC
            mem_dc.BitBlt((0, 0), (width, height), img_dc, (left, top), win32con.SRCCOPY)

            # 将位图保存为文件


            # 将位图对象转换为OpenCV图像
            bmp_info = screenshot.GetInfo()
            bmp_str = screenshot.GetBitmapBits(True)
            img = np.frombuffer(bmp_str, dtype='uint8').reshape((bmp_info['bmHeight'], bmp_info['bmWidth'], 4))

            if filename is not None:
                # 保存位图对象到文件
                img_pil = Image.fromarray(img[..., [2, 1, 0]])
                img_pil.save(filename, format='JPEG', quality=90)

            # 释放对象
            mem_dc.DeleteDC()
            win32gui.DeleteObject(screenshot.GetHandle())
            win32gui.ReleaseDC(hdesktop, desktop_dc)
            #print(time.time()-old_time,img.shape)
            return img



    except :
        print(traceback.format_exc())
        return False



if __name__ == '__main__':
    try:
        step=int(input("多少秒截一张图?输入数字按回车确认:"))
    except:
        step=2
    input("请将游戏设置为1920*1080分辨率,然后按回车开始截图")
    # 获取窗口句柄
    hwnd = win32gui.FindWindow("UnrealWindow", "鸣潮  ")  # 替换成你实际的窗口句柄
    # 设定截图区域的左上角坐标 (x, y) 和右下角坐标 (x, y)
    left, top, right, bottom = 0, 0, 1920, 1080  # 替换成你实际的区域坐标
    #判断是否有这个目录 没有就创建
    if not os.path.exists("./鸣潮截图"):
        os.mkdir("./鸣潮截图")
    while True:
        time.sleep(step)
        filename = f"./鸣潮截图/{int(time.time())}{random.randint(1000,9999)}.jpg"
        ret=screenshot(hwnd, left, top, right, bottom, filename=filename,is_top=False)
        print("截图成功保存在:",filename)













