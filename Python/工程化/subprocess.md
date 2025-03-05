
> subprocess 模块允许我们启动一个新进程，并连接到它们的输入/输出/错误管道，从而获取返回值。

被用来代替 os.system；os.spawn*

与 multiprocessing （同一个代码中通过多进程调用其他的模块）不同，
subprocess 直接调用外部的二进制程序，而非代码模块，**适用于与外部进程交互**。


</br>

## _run_



demo1: 常见用法

```python
import subprocess

cmd = f"ffmpeg -y -ss 00:00:00 -i {video} -to 00:00:08 -c copy video_input.mp4"
subprocess.run(cmd.split())
```



```python
import subprocess

# 执行命令并等待完成
result = subprocess.run(['ls', '-l'], capture_output=True, text=True)

# 输出命令的返回码、标准输出和标准错误
print("Return code:", result.returncode)
print("Standard Output:", result.stdout)
print("Standard Error:", result.stderr)
```

- capture_output=True 和 text=True，可以捕获命令的标准输出和标准错误，并将其作为字符串返回
- shell=True 可以在 shell 中执行命令




</br>

## _check\_output_




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


</br>


## _Popen_


demo3: streamlit

```python
import subprocess

# 在后台启动 Streamlit 服务
def run_streamlit():
    subprocess.Popen(["streamlit", "run", "app.py", "--server.baseUrlPath=front"])
```


Popen 比 run 更加底层，不自动等待执行完成，需要手动 wait 或 communicate;

一般情况使用 run, 流式处理、高级需求使用 Popen


-------------

参考资料：
- https://docs.python.org/zh-cn/3.12/library/subprocess.html

