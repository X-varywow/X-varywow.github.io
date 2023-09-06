
```bash
./ffmpeg

!find . -name '*.mp3' -exec bash -c 'for f; do ffmpeg -i "$f" -acodec pcm_s16le -ar 22050 -ac 1 out/"${f%.mp3}".wav ; done' _ {} +
```

```python
def mp4_to_wav(mp4_path, wav_path, sampling_rate):
    """
    mp4 转 wav
    :param mp4_path: .mp4文件路径
    :param wav_path: .wav文件路径
    :param sampling_rate: 采样率
    :return: .wav文件
    """
    # 如果存在wav_path文件，先删除。
    if os.path.exists(wav_path):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
        os.remove(wav_path)
        # 终端命令
    command = "ffmpeg -i {} -ac 1 -ar {} {} && y".format(mp4_path, sampling_rate, wav_path)
    print('命令是：',command)
    # 执行终端命令
    os.system(command)


if __name__ == '__main__':
    mp4_path = os.getcwd() + r'\record_video.mp4'
    wav_path = os.getcwd() + r'\audio.wav'
    sampling_rate = 16000
    mp4_to_wav("./video_data/tw.mp4", "tw.wav", 16000)
```

trim video to 8 seconds
```bash
ffmpeg -y -ss 00:00:00 -i {video} -to 00:00:08 -c copy video_input.mp4
```





参考资料：
- https://www.ruanyifeng.com/blog/2020/01/ffmpeg.html