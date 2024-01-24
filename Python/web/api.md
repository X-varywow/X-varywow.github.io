
REST, Representational State Transfer，是一种软件架构风格。

- 客户端-服务器架构（CS独立运行，可扩展可移植）
- 无状态（每个请求都包含理解和完成请求所需的全部信息，服务器不存储客户端的会话信息）
- 可缓存性
- 分层系统
- 按需编码
- 统一接口

## _restful api_

一种通用的 API 风格

前置知识：[HTTP](cs/计网/http)


```bash
GET（SELECT）：从服务器取出资源（一项或多项）。
POST（CREATE）：在服务器新建一个资源。
PUT（UPDATE）：在服务器更新资源（客户端提供改变后的完整资源）。
PATCH（UPDATE）：在服务器更新资源（客户端提供改变的属性）。
DELETE（DELETE）：从服务器删除资源。
```


```bash
GET /zoos：列出所有动物园
POST /zoos：新建一个动物园
GET /zoos/ID：获取某个指定动物园的信息
PUT /zoos/ID：更新某个指定动物园的信息（提供该动物园的全部信息）
PATCH /zoos/ID：更新某个指定动物园的信息（提供该动物园的部分信息）
DELETE /zoos/ID：删除某个动物园
GET /zoos/ID/animals：列出某个指定动物园的所有动物
DELETE /zoos/ID/animals/ID：删除某个指定动物园的指定动物
```

携带过滤信息：

```bash
?limit=10：指定返回记录的数量
?offset=10：指定返回记录的开始位置。
?page=2&per_page=100：指定第几页，以及每页的记录数。
?sortby=name&order=asc：指定返回结果按照哪个属性排序，以及排序顺序。
?animal_type_id=1：指定筛选条件
```

状态码：

```bash
200 OK - [GET]：服务器成功返回用户请求的数据，该操作是幂等的（Idempotent）。
201 CREATED - [POST/PUT/PATCH]：用户新建或修改数据成功。
202 Accepted - [*]：表示一个请求已经进入后台排队（异步任务）
204 NO CONTENT - [DELETE]：用户删除数据成功。
400 INVALID REQUEST - [POST/PUT/PATCH]：用户发出的请求有错误，服务器没有进行新建或修改数据的操作，该操作是幂等的。
401 Unauthorized - [*]：表示用户没有权限（令牌、用户名、密码错误）。
403 Forbidden - [*] 表示用户得到授权（与401错误相对），但是访问是被禁止的。
404 NOT FOUND - [*]：用户发出的请求针对的是不存在的记录，服务器没有进行操作，该操作是幂等的。
406 Not Acceptable - [GET]：用户请求的格式不可得（比如用户请求JSON格式，但是只有XML格式）。
410 Gone -[GET]：用户请求的资源被永久删除，且不会再得到的。
422 Unprocesable entity - [POST/PUT/PATCH] 当创建一个对象时，发生一个验证错误。
500 INTERNAL SERVER ERROR - [*]：服务器发生错误，用户将无法判断发出的请求是否成功。
```

</br>

## _接口文档_






</br>

## _api 功能测试_

DEMO, 使用 curl 测试 api:

```bash
curl --location --request POST 'http://you_api/path' \
--header 'authorization: 123' \
--header 'Content-Type: application/json' \
--data-raw '{
    "data": "your_format"
}'
```


DEMO, 使用 python 测试 api:

```python
import requests
import json

data = json.dumps({'col1':1})

response = requests.request("POST", url , headers=headers, data=data)
print(response.text)
```





使用 api 测试工具：apifox, postman 等，会方便很多

https://www.postman.com/

https://apifox.com/


apifox = postman (API开发调试) + swagger（API文档涉及）+ Mock + JMeter（自动化、压力测试）


[apifox 视频教程](https://www.bilibili.com/video/BV1ae4y1y7bf/)

-------

通用规范：

```python
# 请求数据格式
{
    'user_id': 1,
    'name': 'li'
}


# 响应数据格式
{
    'code': 200,
    'message': 'success',
    'data': {
        'col1': 1
    }
}

```

</br>

## _wrk 性能测试_

> Modern HTTP benchmarking tool

https://github.com/wg/wrk

功能测试 -> 性能测试

```bash
sudo apt-get install wrk
```


基本命令格式:

```bash
wrk -t<线程数> -c<连接数> -d<持续时间> <测试URL>
```
其中：
- `-t` 选项指定了要使用的线程数。
- `-c` 选项指定了要打开的HTTP连接数。
- `-d` 选项指定了测试的持续时间，可以用s（秒）、m（分钟）、h（小时）来指定。



示例命令:

```bash
wrk -t12 -c400 -d30s http://127.0.0.1:8080/index.html
```

使用12个线程和400个连接来测试本地服务器上的 "index.html" 页面，测试时间为30秒


!> 生成的是真实的负载，对整个链路的影响较大，即使是测试环境也要注意



-------------

参考资料：
- [性能测试工具 wrk 使用教程](https://www.cnblogs.com/quanxiaoha/p/10661650.html)





--------

参考资料：
- [RESTful API 设计指南](https://www.ruanyifeng.com/blog/2014/05/restful_api.html)
- [REST API 的关键概念](https://mp.weixin.qq.com/s/lRLhXrVN2_-2wVLD4yJ8og)
- https://apifox.com/help/