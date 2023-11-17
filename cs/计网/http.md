> HTTP 是一个 **请求-响应协议**，是两点间传输超文本数据的约定和规范。


> HTTP 是一个 **基于TCP协议的应用层传输协议**。


### 1. HTTP状态码

| 状态码 | 描述                                     |
| ------ | ---------------------------------------- |
| 1XX    | 表示目前是协议处理中间状态               |
| 2XX    | 成功，报文已接受且正确处理               |
| 3XX    | 重定向，资源位置发生变动                 |
| 4XX    | 客户端错误，请求报文有误，服务器无法处理 |
| 5XX    | 服务器处理请求时发生错误                 |

</br>

_常见状态码_

- 422 Unprocessable Entity （检查请求体的格式，如 json 格式，json 中必须用双引号替换单引号）



### 2. HTTP报文

- 请求报文

```bash
GET /admin_ui/rdx/core/images/close.png HTTP/1.1
Accept: */*
Referer: http://xxx.xxx.xxx.xxx/menu/neo
Accept-Language: en-US
User-Agent: Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; WOW64; Trident/7.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET4.0C; .NET4.0E)
Accept-Encoding: gzip, deflate
Host: xxx.xxx.xxx.xxx
Connection: Keep-Alive
Cookie: startupapp=neo; is_cisco_platform=0; rdx_pagination_size=250%20Per%20Page; SESSID=deb31b8eb9ca68a514cf55777744e339
```

- 响应报文

```bash
HTTP/1.1 200 OK
Age: 529651
Cache-Control: max-age=604800
Connection: keep-alive
Content-Encoding: gzip
Content-Length: 648
Content-Type: text/html; charset=UTF-8
Date: Mon, 02 Nov 2020 17:53:39 GMT
Etag: "3147526947+ident+gzip"
Expires: Mon, 09 Nov 2020 17:53:39 GMT
Keep-Alive: timeout=4
Last-Modified: Thu, 17 Oct 2019 07:18:26 GMT
Proxy-Connection: keep-alive
Server: ECS (sjc/16DF)
Vary: Accept-Encoding
X-Cache: HIT

<!doctype html>
<html>
<head>
    <title>Example Domain</title>
	// 省略... 
</body>
</html>
```

### 3. HTTP方法

| HTTP方法  | 作用                             |
| --------- | -------------------------------- |
| `GET`     | 获取资源                         |
| `HEAD`    | 获取报文头部                     |
| `POST`    | 传输实体主体                     |
| `PUT`     | 上传文件，不带验证，一般弃用     |
| `PATCH`   | 对资源部分修改                   |
| `DELETE`  | 删除文件                         |
| `OPTIONS` | 查询支持的方法                   |
| `CONNECT` | 要求在与代理服务器通讯时简历隧道 |
| `TRACE`   | 追踪路径                         |


### 4. HTTP版本

| 版本 | 信息                                                                     |
| ---- | ------------------------------------------------------------------------ |
| 1.0  | 每发起一次请求，都要建立一次TCP链接（三次握手）                          |
| 1.1  | ①提出了长连接通信方式，减少TCP连接开销 。②管道网络传输，减少整体效应时间 |
| 2.0  | ①多路复用 ②首部压缩 ③服务端推送 ④采用二进制格式                          |


> http无状态、明文传输、不安全

HTTP安全问题
- 使用明文进行通信，可能发生窃听
- 不验证通信方的身份，通信方身份可能伪装
- 无法证明报文的完整性，报文可能被篡改



### 5. HTTPS

在TCP和HTTP网络层之间加入了SSL/TLS安全协议
- 功能
  - **加密**（Encryption)
    - 对称密匙加密
    - 非对称密匙加密
    - 混合加密
  - **身份认证**（Authentication)
    - 通过证书来对通信方进行认证
    - 数字证书认证机构（CA）是可信赖的第三方机构
  - **数据一致性** （Data integrity)
    - 通过 MD5 报文摘要
- 缺点
  - 因为加密解密等过程，速度会更慢
  - 需要证书授权费用

HTTP 先和 SSL 通信，再由 SSL 和 TCP 通信，隧道通信

[SSL/TSL证书相关知识](https://juejin.cn/post/7247045258844143674)

### 6. 面试题

!>谈一谈HTTP协议的优缺点

灵活可扩展：语法上只规定了基本格式，传输形式上可传输各种格式数据

请求-应答模式

可靠传输：HTTP基于TCP/IP

无状态：分场景回答，有时如购物系统需要保存用户信息，有时无状态会减少网络开销

明文传输：报文不使用二进制数据，而是文本形式

队头阻塞：当 http 开启长连接时，共用一个TCP连接，当某个请求时间过长时，其它的请求只能处于阻塞状态。

!> 谈一谈 http 各版本之间的差异
    

!> 谈一谈 http 常见状态码

!> 谈一谈 GET 和 POST 的区别




!> DNS 是如何工作的？

DNS 协议提供的是一种主机名到 IP地址 的转换服务，就是我们常说的域名系统。属于应用层协议，通常该协议运行在 UDP 协议之上，使用的是 53 端口号 。

!> DNS 为什么使用 UDP 协议作为传输层协议？

为了避免使用 TCP 协议造成的连接时延

!> 谈一谈正向代理和反向代理

正向代理：我们常说的代理也就是指正向代理，正向代理的过程，它隐藏了真实的请求客户端，服务端不知道真实的客户端是谁，客户端请求的服务都被代理服务器代替来请求。

反向代理：隐藏了真实的服务器，反向代理服务器一般用于负载均衡