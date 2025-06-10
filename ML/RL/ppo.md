
## preface

proximal policy optimization, 近端策略优化。(OpenAI 2017)

基于 Actor-Critic 架构（rl通用架构， actor 策略网络根据当前状态选择动作，critic 价值网络评估动作）， 并尝试保持新旧策略之间的差异适度，限制了策略更新过程中策略变化的幅度，以避免过于激烈的调整导致性能下降。通过截断的策略梯度或裁剪（clipping）约束来实现，提高算法的稳定性。




| 维度             | PPO                                  | DQN                                    |
| ---------------- | ------------------------------------ | -------------------------------------- |
| **算法类型**     | 策略优化（Policy-based）             | 值函数优化（Value-based， Q 函数）     |
| **动作空间**     | 连续或离散                           | 离散为主                               |
| **稳定性**       | 高（剪切或KL约束）                   | 依赖经验回放、目标网络等机制           |
| **样本效率**     | 一般                                 | 较高（重用经验）                       |
| **实现复杂度**   | 中等（比 DQN 稍复杂）                | 简单                                   |
| **适用场景举例** | 机器人臂控制、自动驾驶、复杂策略游戏 | Atari 游戏、简单控制任务、离散策略环境 |


PPO 更适合 连续动作空间（如控制机器人关节角度）或策略变化需要平滑过渡的场景。


## ppo

https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html


```python
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
```





## MaskablePPO

PPO的一个扩展，它允许在训练过程中动态地设置某些动作的掩码，从而屏蔽这些动作。

https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

```bash
pip install sb3-contrib
```

```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
# This is a drop-in replacement for EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback


env = InvalidActionEnvDiscrete(dim=80, n_invalid_actions=60)
model = MaskablePPO("MlpPolicy", env, gamma=0.4, seed=32, verbose=1)
model.learn(5_000)

evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)

model.save("ppo_mask")
del model # remove to demonstrate saving and loading

model = MaskablePPO.load("ppo_mask")

obs, _ = env.reset()
while True:
    # Retrieve current action mask
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, terminated, truncated, info = env.step(action)
```



## Recurrent PPO


https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html




-----------

参考资料：
- [snake-ai](https://github.com/linyiLYi/snake-ai) ; pygame render; MaskablePPO 挺好的demo