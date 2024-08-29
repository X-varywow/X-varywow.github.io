
GCC, general computer control






-------------

（主程）cradle.runner.rdr2_runner.entry

```python
PipelineRunner(llm_provider_config_path=args.llmProviderConfig,
                                    embed_provider_config_path=args.embedProviderConfig,
                                    task_description=task_description,
                                    use_self_reflection = True,
                                    use_task_inference = True)

PipelineRunner.run():

"""
1. 用 json-kv 结构存储 memory, 
关键字段：
    task_description_action, skill_library, exex_info, 
    pre_action, pre_screen_classification, pre_decision_making_reasoning, 
    pre_self_reflection_reasoning
"""
memory.update_info_history(init_params)

"""
2. GameManager,, 
self.ui_control.switch_to_game
raise NotImplementedError； 半成品？
类写的太多，抽象复用继承嵌套下来，感觉并不易读
"""
gm.switch_to_game()

"""
# 3. VideoRecordProvider
threading.Thread(target=capture_screen, daemon=True).start()
input:  mss.mass().grab(region)
output: cv2.VideoWriter()
"""
video_recorder.start_capture()

"""
# 4. VideoClipProvider
memory 应该记录了帧相关的信息， start_frame_id
这里将其帧组成 clip
"""
video_clip(init = True)
gm.pause_game()



# 5. 循环运行
run_information_gathering()
run_self_reflection()
run_task_inference()
run_skill_curation()
run_action_planning()

```

- planner



### information_gathering

```python
# 1. Prepare the parameters to call llm api
self.information_gathering_preprocess()

# 2. Call llm api for information gathering
response = self.information_gathering()

# 3. Postprocess the response
self.information_gathering_postprocess(response)
```

```python
self.planner = RDR2Planner(llm_provider=self.llm_provider,
                            planner_params=config.planner_params,
                            frame_extractor=self.frame_extractor,
                            icon_replacer=self.icon_replacer,
                            object_detector=self.gd_detector,
                            use_self_reflection=True,
                            use_task_inference=True)


InformationGathering(input_map=self.inputs[constants.INFORMATION_GATHERING_MODULE],
                    template=self.templates[constants.INFORMATION_GATHERING_MODULE],
                    text_input_map=self.inputs[constants.INFORMATION_TEXT_GATHERING_MODULE],
                    get_text_template=self.templates[constants.INFORMATION_TEXT_GATHERING_MODULE],
                    frame_extractor=self.frame_extractor,
                    icon_replacer=self.icon_replacer,
                    object_detector=self.object_detector,
                    llm_provider=self.llm_provider)


frame_extractor_gathered_information = None
icon_replacer_gathered_information = None
object_detector_gathered_information = None
llm_description_gathered_information = None
# llm_provider.assemble_prompt(template_str=self.template, params=input)
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


cradle 并不涉及深度的理论知识，全是代码在和 llm 交流。

语言与机器操作这个桥梁的搭建，，
;;;

服了，全是类，函数也写成 class.__call__



----------

参考项目：
- https://github.com/BAAI-Agents/Cradle
- https://github.com/babalae/better-genshin-impact.git
- https://gitee.com/LanRenZhiNeng/ming-chao-ai