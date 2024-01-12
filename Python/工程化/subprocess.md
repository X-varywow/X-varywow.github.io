
> subprocess 模块允许我们启动一个新进程，并连接到它们的输入/输出/错误管道，从而获取返回值。

被用来代替 os.system；os.spawn*

与 multiprocessing （同一个代码中通过多进程调用其他的模块）不同，
subprocess 直接调用外部的二进制程序，而非代码模块，**适用于与外部进程交互**。


demo1: 常见用法

```python
import subprocess

cmd = f"ffmpeg -y -ss 00:00:00 -i {video} -to 00:00:08 -c copy video_input.mp4"
subprocess.run(cmd.split())
```

demo2，附带参数运行命令并返回其输出：

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

如果只需要获取命令的标准输出结果，并且不需要对异常进行处理，可以使用subprocess.check_output()函数。

而如果需要更多的灵活性，并且需要对异常进行处理，可以使用subprocess.run()函数。


demo3: streamlit

```python
import subprocess

# 在后台启动 Streamlit 服务
def run_streamlit():
    subprocess.Popen(["streamlit", "run", "app.py", "--server.baseUrlPath=front"])


# Popen(cmd.split())
```



-------------

参考资料：
- https://docs.python.org/zh-cn/3.11/library/subprocess.html

