import vgamepad as vg

# 左按键
UP = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP
DOWN = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN
LEFT = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT
RIGHT = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT

# ABXY 按键
A = vg.XUSB_BUTTON.XUSB_GAMEPAD_A
B = vg.XUSB_BUTTON.XUSB_GAMEPAD_B
X = vg.XUSB_BUTTON.XUSB_GAMEPAD_X
Y = vg.XUSB_BUTTON.XUSB_GAMEPAD_Y

# 肩键
LB = vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER
RB = vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER

# ?
LT = vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB
RT = vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB

START = vg.XUSB_BUTTON.XUSB_GAMEPAD_START
BACK = vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK


# 左摇杆
def LEFT_JOYSTICK(gamepad, x_value, y_value):
    """
    Args:
        gamepad (_type_): vgamepad.VX360Gamepad
        x_value (_type_): float -1.0 to 1.0
        y_value (_type_): float -1.0 to 1.0
    """
    gamepad.left_joystick_float(x_value, y_value)

def LJ_W(gamepad):
    gamepad.left_joystick_float(0, 1)
def LJ_S(gamepad):
    gamepad.left_joystick_float(0, -1)
def LJ_A(gamepad):
    gamepad.left_joystick_float(-1, 0)
def LJ_D(gamepad):
    gamepad.left_joystick_float(1, 0)

LJ = LEFT_JOYSTICK
# demo: LJ(p1, 0, 1)



# LEFT_JOYSTICK = vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_JOYSTICK
# RIGHT_JOYSTCIK = vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_JOYSTICK
# LEFT_TRIGGER = vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_TRIGGER
# RIGHT_TRIGGER = vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_TRIGGER

# def LEFT_TRIGGER(gamepad, value):
#     gamepad.left_trigger(value)
#     # 左扳机轴 value改成0到255之间的整数
# def RIGHT_TRIGGER(gamepad, value):
#     gamepad.right_trigger(value)
#     # 右扳机轴 value改成0到255之间的整数
# def LEFT_JOYSTICK(gamepad, x_value, y_value):
#     gamepad.left_joystick(x_value, y_value)
#     # 左摇杆XY轴  x_values和y_values 改成-32768到32767之间的整数
# def RIGHT_JOYSTCIK(gamepad, x_value, y_value):
#     gamepad.right_joystick(x_value, y_value)
#     # 右摇杆XY轴  x_values和y_values 改成-32768到32767之间的整数

# def LEFT_TRIGGER(gamepad, value):
#     gamepad.left_trigger_float(value)
#     # 左扳机轴 value改成0.0到1.0之间的浮点值
# def RIGHT_TRIGGER(gamepad, value):
#     gamepad.right_trigger_float(value)
#     # 右扳机轴 value改成0.0到1.0之间的浮点值
# def LEFT_JOYSTICK(gamepad, x_value, y_value):
#     gamepad.left_joystick_float(x_value, y_value)
#     # 左摇杆XY轴  x_values和y_values改成-1.0到1.0之间的浮点值
# def RIGHT_JOYSTCIK(gamepad, x_value, y_value):
#     gamepad.right_joystick_float(x_value, y_value)
#     # 右摇杆XY轴  x_values和y_values改成-1.0到1.0之间的浮点值
