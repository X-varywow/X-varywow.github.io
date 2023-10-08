

MoviePy is a Python library for video editing: cutting, concatenations, title insertions, video compositing (a.k.a. non-linear editing), video processing, and creation of custom effects.



```python
from moviepy.editor import VideoFileClip
import imageio


# 定义输入输出文件名
input_file = "/home/ec2-user/SageMaker/testMilvus/video1.mp4"
output_file= "/home/ec2-user/SageMaker/testMilvus/output.gif"

# 定义输出GIF的目标大小（以字节为单位）
target_size = 200 * 1024  # 200KB

# 加载视频文件
clip = VideoFileClip(input_file)

# 将视频转换为GIF
clip.write_gif(output_file)

# 检查GIF文件大小并进行压缩
gif = imageio.imread(output_file)
while gif.nbytes > target_size:
    clip = clip.resize(0.9)  # 缩小视频尺寸（每次缩小10%）
    
    clip.write_gif(output_file)
    gif = imageio.imread(output_file)
```


------------

参考资料：
- https://github.com/Zulko/moviepy