
> FFmpeg 是视频处理最常用的开源软件。


demo1：显示文件信息
```bash
ffmpeg -version

ffmpeg -i filename
```

</br>

demo2：从图片中合成视频
```bash
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




ffmpeg  -hide_banner \
        -loglevel error \
        -hwaccel auto \
        -r 25.0 \
        -c:v libx264 \
        -crf 6 \
        -pix_fmt yuv420p \
        -vf colorspace=bt709:iall=bt601-6-625 \
        -y temp.mp4



| 参数             | 说明                                |
| ---------------- | ----------------------------------- |
| -hide_banner     | 隐藏无关信息                        |
| -hwaccel auto    | 启用自动硬件加速                    |
| -r 25.0          | 设置输出视频的帧率为25.0帧每秒      |
| -i %04d.jpg      | 指定输入，`%04d`表示4位数的连续序列 |
| -y               | 指定输出                            |
|                  |                                     |
| -c:v libx264     | 使用libx264编码器进行视频压缩       |
| -c:a             | 指定音频编码器                      |
| -crf 6           | 设置压缩质量，数字越小表示质量越高  |
| -pix_fmt yuv420p | 指定输出视频的像素格式为yuv420p     |
| -vf              | 色彩空间转换                        |
|                  |                                     |


</br>

demo3：视频中抽取音频

```bash
ffmpeg -i video.mov -vn -ar 44100 -ac 2 -b:a 128k -f mp3 audio.mp3
```

| 参数 | 说明               |
| ---- | ------------------ |
| -ar  | 采样率             |
| -ac  | 声道数             |
| -f   | 音频格式，自动识别 |
|      |                    |
| -vn  | 视频中抽取音频     |
| -an  | 只抽取视频         |
| -ar  | 采样率             |


------------

参考资料：
- [FFmpeg 视频处理入门教程](https://www.ruanyifeng.com/blog/2020/01/ffmpeg.html)
- 官方文档：https://www.ffmpeg.org/