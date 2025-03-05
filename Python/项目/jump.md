

---------

说明：
- abd + python 基础；
- 纯纯在用色差（rgb 三个的值）在做遍历判定，在这也够用了；


</br>

## _1.环境准备_



#### 1.1 安卓端

以 `MIUI12` 为例，多次点击 miui版本号，进入开发者模式。

以 `HyperOS` 为例，我的设备 -> 多次点击 OS 版本

更多设置 -> 开发者选项中打开USB调试。（可选功能：显示点按操作反馈、指针位置）



#### 1.2 电脑端

去 [android developers](https://developer.android.google.cn/studio/releases/platform-tools) 下载 SDK Platform-Tools

方式1：将其添加到系统的 Path 中，参考：[安装ADB工具包](https://zhidao.baidu.com/question/458972580.html)

方式2：直接在 SDK 文件夹下运行 cmd

然后开启命令行输入 `adb` 即可


------------


```bash
adb devices

adb shell

ls

# 获取手机分辨率
wm size

# 滑动，按压 861ms
input swipe 600.0 2102.1953125 600.0 2102.1953125 861

exit
```


一些踩坑：
- pull: inaccessible or not found； 需要在主机上执行 adb pull, 而不是 adb shell 中执行
- os.system 返回 1 是不正常的情况；在高级设置中添加 adb 路径后需要重启 jupyter


## _2.运行_

在adb指令可执行的情况下，开始游戏，执行该代码即可。

```python
# === 思路 ===
# 核心：每次落稳之后截图，根据截图算出棋子的坐标和下一个块顶面的中点坐标，
#      根据两个点的距离乘以一个时间系数获得长按的时间
# 识别棋子：靠棋子的颜色来识别位置，通过截图发现最下面一行大概是一条直线，就从上往下一行一行遍历，
#         比较颜色（颜色用了一个区间来比较）找到最下面的那一行的所有点，然后求个中点，
#         求好之后再让 Y 轴坐标减小棋子底盘的一半高度从而得到中心点的坐标
# 识别棋盘：靠底色和方块的色差来做，从分数之下的位置开始，一行一行扫描，由于圆形的块最顶上是一条线，
#          方形的上面大概是一个点，所以就用类似识别棋子的做法多识别了几个点求中点，
#          这时候得到了块中点的 X 轴坐标，这时候假设现在棋子在当前块的中心，
#          根据一个通过截图获取的固定的角度来推出中点的 Y 坐标
# 最后：根据两点的坐标算距离乘以系数来获取长按时间（似乎可以直接用 X 轴距离）

#换成银白色皮肤后会失败，黑皮肤正常

# TODO: 解决定位偏移的问题
# TODO: 看看两个块中心到中轴距离是否相同，如果是的话靠这个来判断一下当前超前还是落后，便于矫正
# TODO: 一些固定值根据截图的具体大小计算
# TODO: 直接用 X 轴距离简化逻辑






import os
import shutil
import re
import time
import math
import subprocess
from PIL import Image, ImageDraw
import random
from loguru import logger


# 自己调整的参数
# Magic Number，不设置可能无法正常执行，请根据具体截图从上到下按需设置
under_game_score_y = 300     # 截图中刚好低于分数显示区域的 Y 坐标，300 是 1920x1080 的值，2K 屏、全面屏请根据实际情况修改
piece_half_height = 20         # 二分之一的棋子底座高度，可能要调节
piece_body_width = 70          # 棋子的宽度，比截图中量到的稍微大一点比较安全，可能要调节

press_coefficient = 1.28      # 长按的时间系数，请自己根据实际情况调节。
# （1.4， 1.45）
press_coefficient_x = 1.425


# 全局变量
swipe_x1, swipe_y1, swipe_x2, swipe_y2 = 320, 410, 320, 410     # 模拟按压的起始点坐标，需要自动重复游戏请设置成“再来一局”的坐标
screenshot_backup_dir = './WORK/screenshot_backups/'
tmp_png_path = './WORK/1.png'

scan_start_y = 0


def init_procedure():
    if not os.path.isdir(screenshot_backup_dir):
        os.mkdir(screenshot_backup_dir)

    width, height = 1080, 1920
    res = subprocess.run('adb shell wm size', shell=True, capture_output=True, text=True)
    match = re.search(r'Physical size: (\d+)x(\d+)', res.stdout)
    
    try:
        width = int(match.group(1))
        height = int(match.group(2))
    except:
        logger.warning("No match found")
    return width, height


def pull_screenshot():
    os.system('adb shell screencap -p /sdcard/1.png')
    os.system(f'adb pull /sdcard/1.png {tmp_png_path}')


def backup_screenshot(ts):
    shutil.copy('1.png', '{}{}.png'.format(screenshot_backup_dir, ts))


def save_debug_creenshot(ts, im, piece_x, piece_y, board_x, board_y):
    global scan_start_y
    w, h = im.size

    draw = ImageDraw.Draw(im)

    draw.line((piece_x, piece_y) + (board_x, board_y), fill=2, width=3)

    draw.line((piece_x, 0, piece_x, im.size[1]), fill=(255, 0, 0))
    draw.line((0, piece_y, im.size[0], piece_y), fill=(255, 0, 0))
    draw.line((board_x, 0, board_x, im.size[1]), fill=(0, 0, 255))
    draw.line((0, board_y, im.size[0], board_y), fill=(0, 0, 255))


    draw.line((0, int(h/3),  w, int(h/3)), fill=(0, 0, 0))
    draw.line((0, int(h/3*2),  w, int(h/3*2)), fill=(0, 0, 0))
    draw.line((0, scan_start_y,  w, scan_start_y), fill=(0, 0, 0))


    draw.ellipse((piece_x - 10, piece_y - 10, piece_x + 10, piece_y + 10), fill=(255, 0, 0))
    draw.ellipse((board_x - 10, board_y - 10, board_x + 10, board_y + 10), fill=(0, 0, 255))
    del draw
    im.save("{}{}_d.png".format(screenshot_backup_dir, ts))


def set_button_position(im):
    # 将swipe设置为 `再来一局` 按钮的位置
    global swipe_x1, swipe_y1, swipe_x2, swipe_y2
    w, h = im.size
    left = w / 2      #可改为560（按压的坐标也是再来一局的坐标即可）
    top = 1003 * (h / 1280.0) + 10   #可改为 1648
    swipe_x1, swipe_y1, swipe_x2, swipe_y2 = left, top, left, top


def jump(distance, flag = 0):  #只用考虑时间
    if flag:
        press_time = distance * press_coefficient_x
    else:
        press_time = distance * press_coefficient
    press_time = max(press_time, 200)   # 设置 200 ms 是最小的按压时间
    press_time = int(press_time)   #swipe按压的坐标
    cmd = 'adb shell input swipe {} {} {} {} {}'.format(swipe_x1, swipe_y1, swipe_x2, swipe_y2, press_time)
    # logger.info(cmd)
    os.system(cmd)
    logger.info(f"distance {distance}, press time {press_time}")


def find_piece_and_board(im):
    global scan_start_y
    w, h = im.size

    piece_x_sum = 0
    piece_x_c = 0
    piece_y_max = 0
    board_x = 0
    board_y = 0
    scan_x_border = int(w / 8)  # 扫描棋子时的左右边界
    im_pixel=im.load()
    # 以50px步长，尝试探测scan_start_y
    for i in range(int(h / 3), int( h*2 /3 ), 50):
        last_pixel = im_pixel[0,i]
        for j in range(1, w):
            pixel=im_pixel[j,i]
            # 不是纯色的线，则记录scan_start_y的值，准备跳出循环
            if pixel[0] != last_pixel[0] or pixel[1] != last_pixel[1] or pixel[2] != last_pixel[2]:
                scan_start_y = i - 50
                break
        if scan_start_y:
            break
    logger.info(f"scan_start_y: {scan_start_y}")

    # 从scan_start_y开始往下扫描，棋子应位于屏幕上半部分，这里暂定不超过2/3
    for y in range(int(h / 3), int(h * 2 / 3)):
        for x in range(scan_x_border, w - scan_x_border):  # 横坐标方面也减少了一部分扫描开销
            pixel = im_pixel[x,y]

            # 对棋子颜色定值的判定
            if (50 < pixel[0] < 60) and (53 < pixel[1] < 63) and (95 < pixel[2] < 110):
                piece_x_sum += x
                piece_x_c += 1
                piece_y_max = max(y, piece_y_max)

    if not all((piece_x_sum, piece_x_c)):
        return 0, 0, 0, 0
    piece_x = piece_x_sum / piece_x_c
    piece_y = piece_y_max - piece_half_height  # 上移棋子底盘高度的一半



    # 从上到下逐行扫描；扫描到色差相对大的行时，计算中心点当作目标点；
    for y in range(int(h / 3), int(h * 2 / 3)):
        last_pixel = im_pixel[0, y]
        if board_x:
            break
        board_x_sum = 0
        board_x_c = 0

        for x in range(w):
            pixel = im_pixel[x,y]
            # 修掉脑袋比下一个小格子还高的情况的 bug
            if abs(x - piece_x) < piece_body_width:
                continue

            # 修掉圆顶的时候一条线导致的小 bug，这个颜色判断应该 OK，暂时不提出来
            if abs(pixel[0] - last_pixel[0]) + abs(pixel[1] - last_pixel[1]) + abs(pixel[2] - last_pixel[2]) > 10:
                board_x_sum += x
                board_x_c += 1
        if board_x_sum:
            board_x = board_x_sum / board_x_c
    
    
    # 按实际的角度来算，找到接近下一个 board 中心的坐标 这里的角度应该是30°,值应该是tan 30°, math.sqrt(3) / 3
    board_y = piece_y - abs(board_x - piece_x) * math.sqrt(3) / 3

    # if not all((board_x, board_y)):
    #     return 0, 0, 0, 0

    return piece_x, piece_y, board_x, board_y


def main():
    width, height = init_procedure()
    while True:
        # step1. 从手机截图
        pull_screenshot()
        im = Image.open(tmp_png_path)
        
        # step2. 获取棋子和 board 的位置
        piece_x, piece_y, board_x, board_y = find_piece_and_board(im)
        if not all((piece_x, piece_y, board_x, board_y)):
            logger.warning(f"game end")
            break

        # step3. 跳
        set_button_position(im)
        # jump(math.sqrt((board_x - piece_x) ** 2 + (board_y - piece_y) ** 2))
        jump(abs(board_x - piece_x), flag=1)

        # step4. 保存debug图片
        ts = int(time.time())
        save_debug_creenshot(ts, im, piece_x, piece_y, board_x, board_y)
        # backup_screenshot(ts)

        # print(ts, piece_x, piece_y, board_x, board_y)
        time.sleep(random.uniform(1.5, 1.8))   # 为了保证截图的时候应落稳了，多延迟一会儿


if __name__ == '__main__':
    main()
    # print(init_procedure())
```

</br>

</br>

`2021.3.19` 钢琴块脚本 ⬇

```python
from PIL import Image, ImageDraw
import os
import time

#四个取色的点，也是点击的点
tap=[(90,1200),(360,1200),(630,1200),(900,1200)]


while True:
    os.system('adb shell screencap -p /sdcard/1.png')
    os.system('adb pull /sdcard/1.png .')
    im = Image.open("./1.png")
    im_pixel=im.load()
    
    for x,y in tap:
        color=im_pixel[x,y]
        print(color,x,y)
        if color==(32,32,32,255):  #黑块颜色值为这个
            cmd='adb shell input tap {} {}'.format(x,y)
            print(cmd)
            os.system(cmd)
    
    time.sleep(0.2)  

#复习一下昨天学到的，hhh
#测试环境，小米10pro，钢琴块1，经典模式25块。
#26.477s
#去掉sleep,21.599s。
#好慢啊，用手4.665s
```

## _3.后续_

试着用 opencv 去做：
- canny 边缘检测，不太行，有圆形、四边形，有的形状是一层套一层的，还是个侧视图
- 模板匹配，理论是可行的，但要手动截图，并计算中心点，不能作为一个小demo去做
- 构造数据也比较难构造，因为游戏中相机是移动的，只能以向量形式计算距离。计算出来也不太理想，因为前面检验出的中心点会有偏差


借此机会，又熟悉了一下 ADB, opencv


---------------------

参考资料：
- [github：python跳一跳脚本](https://github.com/wangshub/wechat_jump_game)
- [pillow使用文档](https://pillow.readthedocs.io/en/stable/)
- [python使用adb](https://blog.csdn.net/quikai1981/article/details/78952294)
- [一些adb指令](https://blog.csdn.net/sandalphon4869/article/details/101713495)





