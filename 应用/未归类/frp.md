

frp 穿透

https://github.com/fatedier/frp/releases

公网服务器：入站规则：所有流量都可访问，带个可访问的公网 Ip 即可


```bash
# 查看系统架构，为 x86_64
uname -a
wget https://github.com/fatedier/frp/releases/download/v0.48.0/frp_0.48.0_linux_386.tar.gz
tar -zxvf frp_0.48.0_linux_386.tar.gz

cd frp_0.48.0_linux_386/

# 内网服务器 对应 sagemaker 
vim frpc.ini

# 公网服务器 对应 aws eks
vim frps.ini
```

-------------

另一个内网穿透工具：ngrok ，配置部署相对较简单



-------------

参考：
- https://blog.csdn.net/syxzsyxz1/article/details/121459459
- [frp实现内网穿透以及配置Jupyter Notebook远程连接](https://bingqiangzhou.github.io/2020/06/18/DailyJungle-FrpAndJupyterNotebookRemoteConfig.html)