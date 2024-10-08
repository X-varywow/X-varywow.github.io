
## 1.1 计网基础

>(1) 计算机网络的各层协议

OSI 7层；TCP/IP 4层

- 物理层：实现比特流的透明传输
- 数据链路层：把网络层传下来的数据包组装成 帧
- 网络层：负责数据分组路由与转发
- 传输层：负责进程间的数据传输，即端到端的通信
- 会话层：负责不同机器上用户之间建立及管理会话
- 表示层：用于处理两个通信系统中交换信息的表示方式
- 应用层：对应用程序的通信提供服务

>(2) TCP 和 UDP 的区别

TCP 面向连接，可靠传输，使用流量控制和拥塞控制；面向字节流；适用于文件传输、HTTP；

UDP 无连接，不可靠，不使用流量控制和拥塞控制；面向报文，适用于实时应用；

>(3) 3次握手

①客户端请求建立连接，发送一个同步报文，和初始序列号 seq=x

②服务端收到请求报文，发送**同步确认报文**(SYN=1, ACK=1)，序列号 seq=y，确认号 ack=x+1

③客户端收到服务端的确认后，发送一个**确认报文**（ACK=1），确认号 ack=y+1, 序列号 seq=y+1

>(4) 为什么 TCP 建立连接需要3次握手？

第一次，服务器可确认自己接受正常，对方发送正常；第二次，客户端确认自己接受和对方发送正常，客户端知道都正常；
第三次，服务器知道都正常

>(5) 4次挥手

①客户端发送连接释放报文，主动关闭连接，同时等待确认

②服务端收到连接释放报文，发出确认报文

③服务端发送连接释放报文，主动关闭连接，同时等待确认

④客户端收到连接释放报文，发出确认报文

>(6) 为什么TCP挥手需要4次？

服务器在收到连接释放报文后，可能还有一些数据要传输，不能马上关闭连接，但是会做出应答，返回ACK报文段。

>(7) 为什么四次挥手释放连接时需要等待 2MSL？

MSL 即报文最大生存时间，设置 2MSL 可以保证上一次连接的报文已经在网络中消失，不会出现与新TCP连接报文冲突的情况


>(7) 包丢失

定时器 + 超时重试机制

>(8) TCP 协议如何保证可靠性

可靠传输，即保证接收方收到的字节流和发送方发出的字节流是完全一致的

基本方法：校验和，序列号，滑动窗口，重传机制，拥塞控制，流量控制

参考：[原来 TCP 为了保证可靠传输做了这么多](https://juejin.cn/post/6916073832335802382)

>(9) 如何实现流量控制？

由滑动窗口协议（连续ARQ协议）实现。滑动窗口协议既保证了分组无差错、有序接收，也实现了流量控制。主要的方式就是接收方返回的 ACK 中会包含自己的接收窗口的大小，并且利用大小来控制发送方的数据发送。

通俗：通过滑动窗口就可以实现，接收方告知发送方自己的窗口大小，发送方立刻更改即可实现流量控制。

>(10) 如何实现拥塞控制

（慢开始，逐步增大拥塞窗口的大小，拥塞避免，快重传，快恢复）

>(11) 拥塞控制和流量控制的区别

拥塞控制是作用于网络的，防止过多的数据注入到网络中，避免出现网络负载过大的情况

流量控制是作用于接收方的，它控制发送者的发送速度从而使接收者来得及接收，防止分组丢失

>(12) Cookie 和 Session

Cookie 是服务器发送到用户浏览器并保存在本地的一小块数据，它会在浏览器下次向同一服务器发送请求时携带并发送到服务器上。

Session 代表着服务器和客户端一次会话过程。保存在服务器端

>(13) 简述 DNS 协议

即域名解析系统，是基于 UDP 的应用层协议

首先客户端会在本地缓存服务器中查找这个域名，找不到的话会向根服务器查询，之后根据域名的树状结构一层层递归查询。


>(14) 在浏览器里点击网页，背后发生了什么？

1.浏览器分析www.baidu.com</br>
2.浏览器向DNS服务器请求解析ip地址</br>
3.dns将解析出来的ip地址返回给浏览器</br>
4.三次握手，浏览器与服务器进行tcp连接</br>
5.浏览器向服务器请求html文件</br>
6.服务器返回html文件给浏览器</br>
7.四次挥手，浏览器与服务器断开tcp请求</br>
8.浏览器执行html文件，渲染页面


>(15) select,poll,epoll 了解吗？

select, poll, epoll 是三种 **IO多路复用**（多个socket网络连接或IO流，复用一个线程，被同时监控） 技术。

与多进程和多线程技术相比，I/O多路复用技术的最大优势是系统开销小，系统不必创建进程/线程，也不必维护这些进程/线程，从而大大减小了系统的开销。

select 连接数有限制，基于轮询机制，随着连接数的增加，性能急剧下降。

poll 的实现与 select 非常相似，只是描述 fd 集合的方式不同。

epoll 是目前最好的多路复用技术，连接数无限制，随着连接数增加，性能基本上没有下降。在处理上千万并发连接时，性能很好。（数据结构：红黑树）

个人理解：监听多个IO流并通知的中转器

[python-select](https://docs.python.org/zh-cn/3/library/select.html)

参考：[深入浅出理解select、poll、epoll的实现](https://zhuanlan.zhihu.com/p/367591714)


>(16) ping 命令时，发送的是什么包？

ping 命令执行的时候，源主机首先会构建一个 ICMP 回送请求消息数据包。


>(17) OSPF

即开放最短路径有限协议，使用分布式的链路状态协议，属于网络层协议。

>(18) 了解 Socket 吗？

在操作系统内核空间里，实现网络传输功能的结构是 sock，基于不同的协议和应用场景，会被泛化为不同的sock，它们结合硬件共同实现了网络传输的功能。为了将这部分功能暴露给用户空间的应用程序使用，于是引入了 socket 层，同时将 sock 当成一个特殊的文件，用户可以通过使用 socket fd 句柄来操作内核sock的网络传输能力。

套接字，主要是用来解决网络通信的，计算机进程之间传递消息有时也会用到。

网络套接字是IP地址与端口的组合。

>(19) 列举一些应用层协议

DNS\FTP\HTTP

---------------------------

更多问题：
- [怎么解决TCP网络传输「粘包」问题？](https://zhuanlan.zhihu.com/p/387256713)


## 1.2 HTTP


>(1) get 和 post 的异同

get 和 post 是 HTTP请求的两种方式，都可实现将数据从浏览器向服务器发送带参数的请求。HTTP请求的底层协议都是 TCP/IP

get 用于获取资源，是默认的HTTP请求方法。不会修改服务器的数据，能被缓存，参数保留在浏览器历史中；

post 用于传输实体主体。能修改服务器上的数据，不饿能缓存，参数不会保存在浏览器历史中。

由于get提交的数据放在 URL 中，而post不会，所以get更不太安全。post也不安全，因为HTTP是明文传输，抓包可获取数据内容，要想安全还需要加密。

参考：[GET 对比 POST](https://www.runoob.com/tags/html-httpmethods.html)


>(2) HTTP 协议中常见的请求头和响应头

- 请求头
  - Accept
  - Connection
  - Host
  - Referer
  - User-Agent
  - Cache-Control
  - Cookie
- 响应头
  - Cache-Control
  - Content-Type
  - Content-Encoding
  - Date
  - Server


参考：[关于常用的http请求头以及响应头详解](https://juejin.cn/post/6844903745004765198)


>(3) 说下对HTTP的理解，版本迭代有哪些变化？

HTTP 是一个 请求-响应协议，是两点间传输超文本数据的约定和规范。

HTTP 是一个 基于TCP协议的应用层传输协议。

| 版本 | 信息                                                                     |
| ---- | ------------------------------------------------------------------------ |
| 1.0  | 每发起一次请求，都要建立一次TCP链接（三次握手）                          |
| 1.1  | ①提出了长连接通信方式，减少TCP连接开销 。②管道网络传输，减少整体效应时间 |
| 2.0  | ①多路复用 ②首部压缩 ③服务端推送 ④采用二进制格式                          |


>(4) HTTP 常见状态码

| 状态码 | 描述                                     |
| ------ | ---------------------------------------- |
| 1XX    | 表示目前是协议处理中间状态               |
| 2XX    | 成功，报文已接受且正确处理               |
| 3XX    | 重定向，资源位置发生变动                 |
| 4XX    | 客户端错误，请求报文有误，服务器无法处理 |
| 5XX    | 服务器处理请求时发生错误                 |

参考：[菜鸟教程 - HTTP状态码](https://www.runoob.com/http/http-status-codes.html)

>(5) 谈谈 HTTP首部

HTTP header(HTTP首部)，表示在HTTP请求或响应中的用来传递附加信息的字段

常见首部字段：
- Content-Type: text/html，表示报文主体的对象类型
- Cache-Control，控制缓存的行为


>(6) HTTP如何实现缓存，怎样告诉浏览器这个可以被缓存以及缓存时间

设置 HTTP首部 来实现和控制缓存，像 cache- control，last-modified 这些设置一下就好

>(7) HTTTP 与 HTTPS 的区别

（安全性、开销、端口）

- HTTP 明文传输，数据都是未加密的，安全性较差，HTTPS（SSL+HTTP） 数据传输过程是加密的，安全性较好。
- 使用 HTTPS 协议需要到 CA（Certificate Authority，数字证书认证机构） 申请证书，一般免费证书较少，因而需要一定费用。证书颁发机构如：Symantec、Comodo、GoDaddy 和 GlobalSign 等。
- HTTP 页面响应速度比 HTTPS 快，主要是因为 HTTP 使用 TCP 三次握手建立连接，客户端和服务器需要交换 3 个包，而 HTTPS 除了 TCP 的三个包，还要加上 ssl 握手需要的 9 个包，所以一共是 12 个包。
- http 和 https 使用的是完全不同的连接方式，用的端口也不一样，前者是 80，后者是 443。
- HTTPS 其实就是建构在 **SSL/TLS** 之上的 HTTP 协议，所以，比 HTTP 要更耗费服务器资源。

>(8) 了解 TLS 握手吗？

参考：[TLS 详解握手流程](https://juejin.cn/post/6895624327896432654)

---------------------------------

## 1.3 工程化

>(1) 缓存

加快页面打开速度，减少网络带宽消耗，降低服务器压力

- 分类：
  - 浏览器缓存
  - 页面缓存（将动态页面直接生成的静态页面放在服务器端，用户调取相同页面时，将静态页面直接传到客户端，不再需要程序运行和数据库访问，减少服务器的负载）
  - 数据库缓存（数据库会在内存中划分一个专门的区域，用来存放用户最近执行的查询）
- 应用场景
  - 如图片的预加载


引入缓存可以应对高并发的场景，如抢票、发红包等，系统需要在短时间内完成成千上万次读写操作，不使用缓冲容易造成数据库系统瘫痪，于是就引入了 nosql 技术。redis 和 mongodb 是当前使用最广泛的 nosql 技术。

>(2) 前后端如何通信？

- ajax
- fetch
- websocket(HTML5 提供的一种在单个 TCP 连接上进行全双向通讯的网络协议)
- Form 表单

>(3) 谈谈 Web 框架

参考：
- [Web架构图解](https://blog.csdn.net/weixin_45487120/article/details/124471544)
- [浅谈web网站架构演变过程](https://www.cnblogs.com/xiaoMzjm/p/5223799.html)
- [互联网的前世今生：Web 1.0、2.0、3.0](https://www.cnblogs.com/JasonCeng/p/15861645.html)


-----------------------------


>(4) 了解负载均衡吗？

负载均衡，就是实现集群调度者最优调度解决服务器处理请求的方法。使集群的服务器能够更好地处理用户请求。

现在网站的架构已经从C/S模式转变为B/S模式，现在只需要知道C/S模式是有一个专门的客户端，而B/S模式是将浏览器作为客户端。

实现负载均衡的几种方式：
- HTTP重定向
- DNS负载均衡
- 反向代理负载均衡

----------------------------

>(5) 了解RPC吗？

Remote Procedure Call，即远程过程调用。RPC 可以使用多种传输层协议（TCP UDP等），是一种远程通信的高级抽象，不是一种网络传输协议。

客户端在不知道调用细节的情况下，调用远端服务器暴露出来的方法，就像调用本地方法一样。

常见RPC协议：gRPC，thrift


参考：
- [RPC框架：从原理到选型，一文带你搞懂RPC](https://cloud.tencent.com/developer/article/2021745)
- [聊聊什么是gRPC](https://zhuanlan.zhihu.com/p/363672930)

--------------------------

>(6) 谈谈消息队列

消息队列，是分布式系统中重要的组件。

使用场景：当不需要立即获得结果，但并发量又需要进行控制的时候，如秒杀或抢购活动。
- 限流削峰（即缓冲，对于突发流量，消息队列可保护下游服务）
- 异步处理
- 解耦（上游生产消息，下游处理消息；避免调用接口失败导致整个过程失败）
- 广播（上游生产的消息可被多个下游服务处理）
- 消息驱动



两种模式：
- 点对点模式
- 发布/订阅模式

常用的消息队列：
- Kafka
- Pulsar
- RabbitMQ
- ActiveMQ
- RocketMQ

参考：
- [消息队列及常见消息队列介绍](https://cloud.tencent.com/developer/article/1006035)
- [腾讯技术工程-消息队列](https://mp.weixin.qq.com/s/jWKHAic4Tt4Ohsj4pTmYFw)

-----------------------

>(7) 了解 ZooKeeper 吗？

ZooKeeper主要服务于分布式系统，可以用ZooKeeper来做：统一配置管理、统一命名服务、分布式锁、集群管理。


>(8) 了解微服务吗？

单个应用程序被划分成各种小的、互相连接的微服务，一个微服务完成一个比较单一的功能，相互之间保持独立和解耦合，这就是微服务架构。

hh，这个思想还是 戴森球计划教会我的，拉生产线的时候，单一模块单一功能，是为了：方便修改、扩充产线、方便复制。

就好比你需要1000个蓝色块块（1000个请求）去解锁科技，你只需要扩大对应的子级生产线，但产线杂糅在一起时（服务之间相依性过高），要么开发不易，要么新造一条完全的产线去应对。


>(8) CAP 理论

一个分布式系统最多只能同时满足一致性（Consistency）、可用性（Availability）和分区容错性（Partition tolerance）这三项中的两项。

> 什么是Docker？

参考：https://zhuanlan.zhihu.com/p/187505981

## 1.4 安全相关

>(1) sql注入是怎么产生的，如何防止？

因为在程序开发过程中没有对sql语句进行检查或未进行关键字检查， 导致客户端可以提交sql语句到服务器运行。

如何防止：
- 对sql与进行检查， 过滤。
- 不要使用sql拼接的方式来拼接sql语句， 对常用的方法进行封装避免暴露sql语句（使用ORM）。

>(2) xss是什么？如何防范?

xss，即跨站脚本攻击，通过利用网页开发时留下的漏洞，将恶意指令代码注入到网页中，使用户加载并执行攻击者恶意制造的网页程序。

如何防范？
- 不信任用户提交的任何内容，对所有用户提交内容进行可靠的输入验证
- 实现Session 标记（session tokens）、CAPTCHA（验证码）系统或者HTTP引用头检查
- cookie 防盗。避免直接在cookie中泄露用户隐私。
- 确认接收的内容被妥善地规范化，仅包含最小的、安全的Tag（没有JavaScript），去掉任何对远程内容的引用（尤其是样式表和JavaScript），使用HTTPonly的cookie。


>(3) csrf是什么？如何防范？

cross site request forgery，即跨站伪造请求， 利用用户信任过的网站去执行一些恶意的操作

如何防范：
- 检查 HTTP Referer字段；严格要求该字段只来自于信任的URL
- 在请求地址中添加 token 并验证；


参考：[浅谈CSRF攻击方式](https://www.cnblogs.com/hyddd/archive/2009/04/09/1432744.html)

>(4) 其它攻击方式

DDOS：它在短时间内发起大量请求，耗尽服务器的资源，无法响应正常的访问。[参考](http://www.ruanyifeng.com/blog/2018/06/ddos.html)

弱口令：仅包含简单数字和字母的口令。通过提高口令强度防止可改进。
