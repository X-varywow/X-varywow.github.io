
> facefusion 与 roop 文件结构等几乎一致，应该是同一作者。</br></br>
> 这两者都是都过一张图片作为参考图片，不需要数据集和训练，当然效果也没那么好


- [x] facefusion 已在 windows 本地测试良好，cuda 11.8；效果根据素材情况而定，很难较好
- [x] roop 在 colab 跑通
- [ ] sagemaker 各种问题
- [ ] mac 部分通的，会卡住，，，





## roop

仓库地址：https://github.com/s0md3v/roop

https://github.com/dream80/roop_colab

coalb 使用这个可以直接跑通：[colab 使用参考](https://github.com/dream80/roop_colab/blob/main/roop_v1_3.ipynb)


> 直接在 sagemaker 里 run.py 不行， 是基于 tkinter 构建的；facefusion 基于 gradio webui



----------


报错：[ROOP.FACE-SWAPPER] No face in source path detected.

查看代码，替换人脸检测部分，对于卡通脸检测的有问题，

> 文档啥都没有，也没有论文。

```bash
# 安装 ffmpeg, 路径正确

conda create -n roop python=3.10 -y
source activate roop

cd roop
pip install -r requirements


python run.py --execution-provider cuda -s ./assets/img1.png -t ./assets/video1.mp4 -o ./out.mp4 --frame-processor face_swapper face_enhancer --output-video-encoder libx264 --output-video-quality 35 --keep-fps    --temp-frame-format png --temp-frame-quality 0
```

> 跑得比较慢(1s一分钟)，显存也没有吃满；加速也加了，onnxruntime-gpu cuda 11.8；

> 是真的慢，远不能达到实时的要求

跑了两遍，什么 bug 都没有保存下来。。。


## facefusion

最新的换脸方案

仓库地址：https://github.com/facefusion/facefusion


没有相关论文

https://docs.facefusion.io/installation 文档其实啥都没有


[facefusion model download](https://huggingface.co/facefusion)





## 报错杂记


安装环境报错：
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
triton 2.0.0 requires cmake, which is not installed.
triton 2.0.0 requires lit, which is not installed.
pydantic-core 2.3.0 requires typing-extensions!=4.7.0,>=4.6.0, but you have typing-extensions 4.5.0 which is incompatible.

运行报错：
/lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found 

系统版本或静态链接过程出问题，报错报麻了，算了


> roop 没有保存的文件，试试 facefusion

就是报错，啊啊


同样问题，有了报错：[FACEFUSION.CORE] Creating video failed， 可以定位了

```python
update_status(wording.get('creating_video_fps').format(fps = fps))

if not create_video(facefusion.globals.target_path, fps):
    print("creating_video_failed")
		return
```


```python
def create_video(target_path : str, fps : float) -> bool:
	temp_output_path = get_temp_output_path(target_path)
	temp_directory_path = get_temp_directory_path(target_path)
	output_video_quality = round(51 - (facefusion.globals.output_video_quality * 0.5))

	commands = [ '-hwaccel', 'auto', '-r', str(fps), '-i', os.path.join(temp_directory_path, '%04d.' + facefusion.globals.temp_frame_format), '-c:v', facefusion.globals.output_video_encoder ]
	if facefusion.globals.output_video_encoder in [ 'libx264', 'libx265', 'libvpx' ]:
		commands.extend([ '-crf', str(output_video_quality) ])
	if facefusion.globals.output_video_encoder in [ 'h264_nvenc', 'hevc_nvenc' ]:
		commands.extend([ '-cq', str(output_video_quality) ])

	commands.extend([ '-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625', '-y', temp_output_path ])
	print(commands)
	return run_ffmpeg(commands)

# commands
# ['-hwaccel', 'auto', '-r', '25.0', '-i', '/tmp/facefusion/3-4x5/%04d.jpg', '-c:v', 'libx264', '-crf', '6', '-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625', '-y', '/tmp/facefusion/3-4x5/temp.mp4']
```

```python
def run_ffmpeg(args : List[str]) -> bool:
	commands = [ 'ffmpeg', '-hide_banner', '-loglevel', 'error' ]
	commands.extend(args)
	try:
		subprocess.check_output(commands, stderr = subprocess.STDOUT)
		return True
	except subprocess.CalledProcessError:
		return False
```

```bash
# 最终运行的 ffmpeg ：
ffmpeg  -hide_banner \
        -loglevel error \
        -hwaccel auto \
        -r 25.0 \
        -i /tmp/facefusion/3-4x5/%04d.jpg \
        -c:v libx264 \
        -crf 6 \
        -pix_fmt yuv420p \
        -vf colorspace=bt709:iall=bt601-6-625 \
        -y /tmp/facefusion/3-4x5/temp.mp4
```

```bash
# 测试
ffmpeg  -hide_banner \
        -hwaccel auto \
        -r 25.0 \
        -i /home/ec2-user/SageMaker/facefusion/tmp/facefusion/1-4x5/%04d.jpg \
        -pix_fmt yuv420p \
        -vf colorspace=bt709:iall=bt601-6-625 \
        -y /home/ec2-user/SageMaker/facefusion/tmp/facefusion/1-4x5/temp.mp4
```

```bash
# 测试
ffmpeg  -hide_banner \
        -hwaccel auto \
        -r 25.0 \
        -i /home/ec2-user/SageMaker/facefusion/tmp/facefusion/2-4x5/%04d.jpg \
        -y /home/ec2-user/SageMaker/facefusion/tmp/facefusion/2-4x5/temp.mp4
```

这一堆 ffmpeg 参数这么多，把 ffmpeg 抽出来就有报错信息了。

麻了，sagemaker 就各种报错，挑出隐藏的 ffmpeg 语句，改变 -crf ，后面又是整个 encoder 不行

> sagemaker 就是脏水里洗衣服，不知道什么鬼系统，各种问题。colab ffmpeg 是预装的

crf 是 x264编码器的一个选项，用于指定视频的质量。

那个 tmp 一关就会消失，又是新的代码

对于卡通脸，一般都是检测不到的；男脸2，检测得到；


- 绿色、紫色条纹 ？？ 色彩空间又出问题了
