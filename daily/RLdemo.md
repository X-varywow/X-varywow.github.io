

yolox + dqn

还得是算力

## YOLOX




## autofish

- fisher (定义 DQN 网络中 agent environment model predictor)
- utils (键鼠操作、画图)




## cv 识图


```python
def psnr(img1, img2):
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
```

## win32api

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

## win32gui

获取窗口、句柄信息

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

## DQN 网络

### model

有点简陋啊,,,

```python
# fishnet
nn.Linear(in_ch, 16),
nn.LeakyReLU(),
nn.Linear(16, out_ch)

# movefishnet
nn.Linear(in_ch, 32),
nn.LeakyReLU(),
nn.Linear(32, 32),
nn.LeakyReLU(),
nn.Linear(32, out_ch)
```

### 定义环境

使用的 opencv 

找到对应图片的位置、


### agent

看起来也简陋

有趣

```python
    def choose_action(self, x):
        # This function is used to make decision based upon epsilon greedy
        x = torch.FloatTensor(x).unsqueeze(0)  # add 1 dimension to input state x
        # input only one sample
        if np.random.uniform() < self.epsilon:  # greedy
            # use epsilon-greedy approach to take action
            actions_value = self.eval_net.forward(x)
            # torch.max() returns a tensor composed of max value along the axis=dim and corresponding index
            # what we need is the index in this function, representing the action of cart.
            action = actions_value if self.reg else torch.argmax(actions_value, dim=1).numpy()  # return the argmax index
        else:  # random
            action = np.random.rand(self.n_actions)*2-1 if self.reg else np.random.randint(0, self.n_actions)
        return action
```

## RL 理论


DDQN（Double Deep Q-Network）是DQN（Deep Q-Network）的一种改进算法。

主要的区别在于DDQN在计算目标Q值时使用了两个网络，即一个主要网络用于选择动作，另一个目标网络用于计算目标Q值。