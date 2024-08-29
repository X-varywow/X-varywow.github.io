
GCC, general computer control






-------------

（主程）cradle.runner.rdr2_runner.entry

```python
PipelineRunner(llm_provider_config_path=args.llmProviderConfig,
                                    embed_provider_config_path=args.embedProviderConfig,
                                    task_description=task_description,
                                    use_self_reflection = True,
                                    use_task_inference = True)
```






-------------

(子步骤) cv_go_to_icon 

1. Get observation screenshot

```python
import mss

with mss.mss() as sct:
    screen_image = sct.grab(region)
    image = Image.frombytes("RGB", screen_image.size, screen_image.bgra, "raw", "BGRX")
    image.save(screen_image_filename)
```

config 中配置截图位置截取

2. match_template

```python
from MTM import matchTemplates

detection = matchTemplates([('', cv2.resize(template, (0, 0), fx=s, fy=s)) for s in [1]],
                            srcimg,
                            N_object=1,
                            method=cv2.TM_CCOEFF_NORMED,
                            maxOverlap=0.1)

# match_template 还会给出 template 与当前的置信度、角度、距离
```

3. move
```python
# record prev_dis, prev_theta to check stuck
# stuck 的处理怪怪的
for _ in range(2):
    turn(np.random.randint(30, 60) if np.random.rand()<0.5 else -np.random.randint(30, 60))
    move_forward(np.random.randint(2, 4))

# 3. Move
turn(theta)                     # 移动鼠标
move_forward(1.5)               # key_hold('w', duration)
time.sleep(0.5)

if debug:
    logger.debug(f"step {step:03d} | timestep {timestep} done | theta: {theta:.2f} | distance: {dis:.2f} | confidence: {confidence:.3f} {'below threshold' if confidence < 0.5 else ''}")

prev_dis, prev_theta = dis, theta
```



------------



- [ ] cardle rdr2





----------

参考项目：
- https://github.com/BAAI-Agents/Cradle
- https://github.com/babalae/better-genshin-impact.git
- https://gitee.com/LanRenZhiNeng/ming-chao-ai