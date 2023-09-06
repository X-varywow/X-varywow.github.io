
## target

做个 ydzy， ppo

## preface

?>gym 已经被移入到 gymnasium

```python
import gymnasium as gym
```

Gym is a standard API for reinforcement learning, and a diverse collection of reference environments


[Gymnasium 文档](https://gymnasium.farama.org/)  

[gym 文档](https://www.gymlibrary.dev/)

https://www.gymlibrary.dev/content/basic_usage/


```python
!pip install gym
!pip install gym[box2d]
```

```python
import gym
env = gym.make("LunarLander-v2", render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
#     print(observation, reward, terminated, truncated, info)
#              list(8),    1.1,      false,    false,     {}


    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

一个完整的过程，通常叫 episod，整个生命周期的奖励 $R = \sum_{t=1}^Tr_t$


- on plicy，训练数据由当前 agent 不断环境交互得到
- off plicy，找个代练


---------

参考资料：
- https://www.bilibili.com/video/BV1yP4y1X7xF?p=7

其他项目：
- [snake-ai](https://github.com/linyiLYi/snake-ai)

