"""
在线测试手柄：https://www.9slab.com/gamepad/home

https://github.com/yannbouteiller/vgamepad
"""


import keyboard
import vgamepad
from loguru import logger

from tmp.conf import *

p1 = vgamepad.VX360Gamepad()
p2 = vgamepad.VX360Gamepad()

single_flag = False


joystick_map = {
    "w": (12, LJ_W),
    "s": (12, LJ_S),
    "a": (12, LJ_A),
    "d": (12, LJ_D),

    "up": (2, LJ_W),
    "down": (2, LJ_S),
    "left": (2, LJ_A),
    "right": (2, LJ_D),
}

key_map = joystick_map | {
    # # 双人移动控制
    # "w": (12, UP),
    # "s": (12, DOWN),
    # "a": (12, LEFT),
    # "d": (12, RIGHT),

    # # p2 单独控制
    # "up": (2, UP),
    # "down": (2, DOWN),
    # "left": (2, LEFT),
    # "right": (2, RIGHT),

    # DPAD
    "1": (12, UP),
    "2": (12, DOWN),
    "3": (12, LEFT),
    "4": (12, RIGHT),

    # ABXY 按键
    "h": (1, A),
    "j": (2, A),
    "space": (12, B),
    "b": (12, B),
    "x": (12, X),
    "y": (12, Y),

    "tab": START,
    "esc": BACK,
}


def on_key_event(event):
    # print(event)
    global single_flag
    if event.event_type == keyboard.KEY_DOWN and event.name == "f6":
        logger.info(f"single_flag: {single_flag}")
        single_flag = not single_flag
        return

    if event.name not in key_map: return
    
    players, xbox_key = key_map[event.name]
    if single_flag:
        players = [p1]
    elif players == 12:
        players = [p1, p2]
    elif players == 1:
        players = [p1]
    else:
        players = [p2]


    for player in players:
        if xbox_key in [LJ_W, LJ_S, LJ_A, LJ_D]:
            xbox_key(player)
        else:
            player.press_button(xbox_key) if event.event_type == keyboard.KEY_DOWN else player.release_button(xbox_key)
        logger.info(f"player: {player} release xbox_key: {xbox_key}")

    p1.update()
    p2.update()


keyboard.hook(on_key_event)

# 保持脚本运行
keyboard.wait('f5')  # 按f5键退出脚本keyboard.wait('f5')  # 按f5键退出脚本