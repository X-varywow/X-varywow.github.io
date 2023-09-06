


## imageio

[官方文档](https://imageio.readthedocs.io/en/stable/user_guide/index.html)


demo1: 视频读取

```python
import imageio.v3 as iio
import numpy as np

# read a single frame
# 视频中存在 0 帧
frame = iio.imread(
    "imageio:cockatoo.mp4",
    index=0,
    plugin="pyav",
)

print(type(frame), frame.shape) # <class 'numpy.ndarray'> (720, 1280, 3)
```

```python
# bulk read all frames
# Warning: large videos will consume a lot of memory (RAM)
frames = iio.imread("imageio:cockatoo.mp4", plugin="pyav")       # <class 'numpy.ndarray'>

# iterate over large videos
for frame in iio.imiter("imageio:cockatoo.mp4", plugin="pyav"):  # <class 'generator'>
    print(frame.shape, frame.dtype)
```


> 使用 pyav 作为后端会比默认的 imageio-ffmpeg 更快

```python
%timeit iio.imread("imageio:cockatoo.mp4", index=2, plugin="pyav")

%timeit iio.imread("imageio:cockatoo.mp4", index=2)

# 47.2 ms ± 1.11 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 77.4 ms ± 238 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

demo2: 读取多个特定帧

```python
# way1

index = [1,2,3,4,5,6,7,8]
img_list = []
for i, frame in enumerate(iio.imiter("imageio:cockatoo.mp4")):
    if i in index:
        img_list.append(frame)

img_array = np.asarray(img_list)
```

```python
# way2

index = [1,2,3,4,5,6,7,8]

with iio.imopen("imageio:cockatoo.mp4", "r", plugin="pyav") as img_file:
    img_list = [img_file.read(index=idx) for idx in index]

img_array = np.stack(img_list)
```







demo3：从视频中读取图片，进行风格转换

```python
def inference(filepath, style):
    now = time.time()
    
    outpath = f'output_{str(now)[:-8]}.mp4'

    reader = imageio.get_reader(filepath)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(outpath, mode='I', fps=fps, codec='libx264')

    model_name = model_dict[style]
    img_cartoon = pipeline(Tasks.image_portrait_stylization, model=model_name)

    for _, img in tqdm(enumerate(reader)):
        result = img_cartoon(img[..., ::-1])
        res = result[OutputKeys.OUTPUT_IMG]
        writer.append_data(res[..., ::-1].astype(np.uint8))
    writer.close()
    print('finished!')

    return outpath
```





参考资料：
- [使用Python进行基本图像数据分析](https://www.51cto.com/article/582460.html)
- [官方教程](https://imageio.readthedocs.io/en/stable/examples.html#read-or-iterate-frames-in-a-video)