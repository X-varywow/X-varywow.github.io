
## preface

proximal policy optimization, 近端策略优化。(OpenAI 2017)

PPO的主要特点是尝试保持新旧策略之间的差异适度，即它限制了策略更新过程中策略变化的幅度以避免过于激烈的调整导致性能下降。它通过截断的策略梯度或裁剪（clipping）约束来实现。

具体地，PPO定义了一个“剪切”目标函数，用来维持两个连续策略之间的一个预设界限，如果实际的策略变动超过这个界限，该目标函数会限制该变动，从而提高算法的稳定性。




## ppo




## MaskablePPO

PPO的一个扩展，它允许在训练过程中动态地设置某些动作的掩码，从而屏蔽这些动作。


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

https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
