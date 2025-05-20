


## preface


Gym is a standard API for reinforcement learning, and a diverse collection of reference environments



```bash
pip install gymnasium
pip install swig
pip install "gymnasium[box2d]"
```


## base

demo: 控制物体降落到合适位置，一次降落具有多帧（多个连续的 action, obs ...）

```python
import gymnasium as gym

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)

for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)
#     print(observation, reward, terminated, truncated, info)
#              list(8),    1.1,      false,    false,     {}

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```




## envs

```python
env = gym.make("LunarLander-v3", render_mode="human")

```


### 内置 env





### 自定义 env


## action



---------

参考资料：
- https://www.bilibili.com/video/BV1yP4y1X7xF?p=7
- https://gymnasium.farama.org/content/basic_usage/

其他项目：
- [snake-ai](https://github.com/linyiLYi/snake-ai)

