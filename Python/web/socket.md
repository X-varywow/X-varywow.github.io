

## _preface_

>Socket又称"套接字"，应用程序通常通过"套接字"向网络发出请求或者应答网络请求，使主机间或者一台计算机上的进程间可以通讯。


Python 提供的网络服务：
- 低级别的网络服务支持基本的 Socket，它提供了标准的 BSD Sockets API，可以访问底层操作系统 Socket 接口的全部方法。
- 高级别的网络服务模块 SocketServer， 它提供了服务器中心类，可以简化网络服务器的开发。


```python
import socket

# 创建一个socket:
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 建立连接:
s.connect(('www.sina.com.cn', 80))

# 发送数据:
s.send(b'GET / HTTP/1.1\r\nHost: www.sina.com.cn\r\nConnection: close\r\n\r\n')

# 接收数据:
buffer = []
while True:
    # 每次最多接收1k字节:
    d = s.recv(1024)
    if d:
        buffer.append(d)
    else:
        break
data = b''.join(buffer)

# 关闭连接:
s.close()
```


## _demo_

python 实现curl: https://github.com/michellcampos/pythoncccurl/blob/main/main.py

```python
import socket
import argparse
from urllib.parse import urlparse


def build_request(method, urlparsed, header=None, data=None) -> str:
    request_header = f"{method} {urlparsed.path} HTTP/1.1\r\n"
    request_header += f"Host: {urlparsed.hostname}\r\n"
    request_header += "User-Agent: curl/8.1.2\r\n"
    request_header += "Accept: */*\r\n"
    request_header += "Connection: close\r\n"

    if method in ['POST', 'PUT', 'PATCH']:
        if header:
            for h in header:
                request_header += f"{h}\r\n"

        if data:
            request_header += f"Content-Length: {len(data)}\r\n\r\n"
            request_body = f"{data}\r\n\r\n"

    return request_header + request_body if method in ['POST', 'PUT', 'PATCH'] else request_header + "\r\n"


def send_request(urlparsed, request_header) -> bytes:
    port = urlparsed.port if urlparsed.port else 80

    with socket.create_connection((urlparsed.hostname, port)) as sock:
        sock.sendall(request_header.encode())

        response = b""
        while True:
            data = sock.recv(4096)
            if not data:
                break
            response += data

    return response


def parse_response(request_header, response, verbose=False) -> None:
    response_splited = response.split(b'\r\n\r\n')
    if verbose:
        for line in request_header.splitlines():
            print(f'> {line}')

        for line in response_splited[0].splitlines():
            print(f'< {line}')

        print(f'<\r\n{response_splited[1].decode()}')

    else:
        print(response_splited[1].decode())


def main() -> None:
    parser = argparse.ArgumentParser(description='Curl-like HTTP client')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-X', '--method', default='GET', help='HTTP method')
    parser.add_argument('url', help='URL to send the request to')
    parser.add_argument('-d', '--data', help='Data to send in the request body')
    parser.add_argument('-H', '--header', action='append', help='Additional headers')

    args = parser.parse_args()

    verbose = args.verbose
    method = args.method
    url = args.url
    data = args.data
    header = args.header

    urlparsed = urlparse(url)

    request_header = build_request(method, urlparsed, header, data)
    response = send_request(urlparsed, request_header)
    parse_response(request_header, response, verbose)


if __name__ == "__main__":
    main()
```


------------------

参考资料：
- https://www.runoob.com/python/python-socket.html
- [TCP编程-廖雪峰](https://www.liaoxuefeng.com/wiki/1016959663602400/1017788916649408)