


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


### 内置 env

```python
env = gym.make("LunarLander-v3", render_mode="human")

```




### 自定义 env


## spaces

环境需包含：`action_space`, `observation_space`

```python
# {0,1,2}
# two-dimensional unit cube
self.action_space = spaces.Discrete(3)
self.observation_space = spaces.Box(0, 1, shape=(2,))
```



## action



---------

参考资料：
- https://www.bilibili.com/video/BV1yP4y1X7xF?p=7
- https://gymnasium.farama.org/content/basic_usage/
