
一些项目中会有如下代码：

```python
# monkey patch ssl
if platform.system().lower() == 'darwin':
	ssl._create_default_https_context = ssl._create_unverified_context
```

macos 系统默认会验证 https 请求的证书。

开发过程中，使用 ssl._create_unverified_context 实现禁用证书验证。