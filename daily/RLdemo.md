

yolox + dqn

还得是算力

图像表示，有效特征提取，就算非常难了，内存提取

### YOLOX

用于鱼的定位、类型识别、鱼竿落点定位，这里放弃传统了传统的 opencv 方法。

研究一下差异，

这样的话，特征提取靠谱多了，自动化探索


## repo1 autofish

- fisher (定义 DQN 网络中 agent environment model predictor)
- utils (键鼠操作、画图)




### cv 识图


```python
def psnr(img1, img2):
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
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

### RL 理论


DDQN（Double Deep Q-Network）是DQN（Deep Q-Network）的一种改进算法。

主要的区别在于DDQN在计算目标Q值时使用了两个网络，即一个主要网络用于选择动作，另一个目标网络用于计算目标Q值。




## repo2 assistnt

https://github.com/infstellar/genshin_impact_assistant.git
https://github.com/PyQt5/PyQt

弄个 git;
agent + local memory
memory 共享