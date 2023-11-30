

## _穿透_

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



</br>

## _隧道_

网络中的隧道，可以让数据安全地在不同网络之间传输。

数据包封装在其他协议中传送，与外界隔离，确保数据可以跨越不兼容的网络、安全地通过不可信的网络，或绕过某些网络限制。

隧道常用于虚拟私人网络（VPN）中，以确保跨公共网络的通信具有隐私性和安全性。

常见的隧道协议包括：PPTP，L2TP, IPSec 和 SSL/TLS


-----------

使用 ssh -L 可建立简单的隧道连接

```bash
ssh -L [本地端口]:[目标主机]:[目标端口] [用户]@[远程服务器]

ssh -L 1234:localhost:3306 user@prod.server.com

ssh -L 3303:localhost:3306 root@192.168.1.104
```

如果在退出 SSHsession 后保持通道运行，可以添加 -N 不执行远程命令 和 -f 后台运行




-------------

参考资料：
- https://blog.csdn.net/syxzsyxz1/article/details/121459459
- [frp实现内网穿透以及配置Jupyter Notebook远程连接](https://bingqiangzhou.github.io/2020/06/18/DailyJungle-FrpAndJupyterNotebookRemoteConfig.html)
- [linux下建立ssh tunnel实现端口转发](https://blog.csdn.net/beeworkshop/article/details/99711504)
- chatgpt