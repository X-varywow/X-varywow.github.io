
## Preface

> `2022.03.31` 图片上传工具 picgo 出 BUG 打不开，picgo还无法管理COS，picgo 还有时自动清空配置；<br>
> 所以，这里，用 python 写了一个 图片上传、腾讯云COS管理的工具

需求分析：
- 本地的图片，自动按时间重命名，并上传到 COS


实现思路：
- 仿照 git 设置三个地点（管理方便）
  - 地点1：初始地点，图片未重命名
  - 地点2：本地暂存地点，图片重命名，一旦进入就自动上传到地点3
  - 地点3：腾讯云 COS 

## 代码实现

```python
# -*- coding:utf-8 -*-
# Filename: CosUploader
# Author：jxq
# Date: 2022-03-31

# 主要功能说明（自用）：
# client.up() 用于将 地点1 的所有文件，重命名到地点2，并上传到 web
# client.sync_local_to_web() 用于本地删除图片后，同步到 web 端删除

#  pip install -U cos-python-sdk-v5

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import sys
import os
import shutil
import logging
import time
from pprint import pprint

# 正常情况日志级别使用INFO，需要定位时可以修改为DEBUG，此时SDK会打印和服务端的通信信息
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

class CosUploader():
    def __init__(self):
        self.secret_id = ''
        self.secret_key = ''
        self.bucket = ''
        self.region = 'ap-beijing'
        self.token = None
        self.scheme = 'https'
        self.config = CosConfig(Region=self.region, SecretId=self.secret_id, SecretKey=self.secret_key, Token=self.token, Scheme=self.scheme)
        self.client = CosS3Client(self.config)
        self.path1 = "C:\\Users\\Administrator\\Pictures\\"
        self.path2 = "C:\\Users\\Administrator\\Pictures\\cos_pic\\"
    
    # 返回 桶名 列表
    def ls_bk(self):
        data = self.client.list_buckets()['Buckets']['Bucket']
        res = [d['Name'] for d in data]
        return res
    
    # 下载 ls_pic 的所有图片到 本地 path2
    def download_all(self):
        cnt = 1
        for name, size in self.ls_pic():
            response = self.client.get_object(Bucket=self.bucket, Key=name)
            response['Body'].get_stream_to_file(self.path2 + name)
            print("第{}张传输完成".format(cnt))
            cnt += 1
        print("Success!")
    
    # 打印并返回 图片 统计信息,
    # mod 为 1 时打印全部信息
    def ls_pic(self, mod = 0):
        data = self.client.list_objects(Bucket=self.bucket)
        res = [(d['Key'], d['Size']) for d in data['Contents']]
        
        cnt, size = 0, 0
        for name, s in res:
            if mod:
                print("图片名：{}，\t大小：{:.2f}kb".format(name, int(s)/1000))
            cnt += 1
            size += int(s)
        
        print("【web端图片】总数量：{}，总大小：{:d}mb".format(cnt, size//1000000))
        # return data
        return res
    
    
    # 二进制，上传单个文件
    def upload(self, img_path, name):
        with open(img_path, 'rb') as fp:
            response = self.client.put_object(
            Bucket=self.bucket,
            Body=fp,
            Key=name,
            StorageClass='STANDARD',
            EnableMD5=False)
        #print(response['ETag'])
    
    # 最重要的功能，名字就用 up 了
    # 地点1 （重命名）-> 地点2（上传）-> 地点3
    def up(self):
        skip = ['desktop.ini', 'up.bat','CosUploader.py'] # 跳过上传的文件
        key = time.strftime("%Y%m%d%H%M%S", time.localtime())
        
        for file in os.listdir(self.path1):
            old_name = self.path1+file
            
            if os.path.isfile(old_name) and file not in skip:
                print(file)
                key = time.strftime("%Y%m%d%H%M%S", time.localtime())
                suffix = file.split(".")[-1]
                new_name = self.path1 + key + "." + suffix
                dstfile = self.path2 + key + "." + suffix
                
                os.rename(old_name,new_name)
                shutil.move(new_name,dstfile) 
                print("Success:{}->{}".format(old_name, dstfile))
                
                self.upload(dstfile, key + "." + suffix)
                print("Success:{}->腾讯云COS".format(dstfile))
            
            time.sleep(1) # 防止名字重了
    
    
    # 一般用于：本地删除图片，之后同步到 web 端的删除
    # 只用于删除 web 图片，本地 up 包含了 sync 
    def sync_local_to_web(self):
        local_files = os.listdir(self.path2)
        web_files = [i[0] for i in self.ls_pic()]
        cnt = 0

        for file in web_files:
            if file not in local_files:
                response = self.client.delete_object(Bucket='***',Key=file)
                print("已删除：{}".format(file))
                cnt += 1

        if not cnt:
            print("本地和 web 端一致，没有删除图片")
            
    
    
    # 这个一般不会用，用 download_all 就行了
    def sync_web_to_local(self):
        pass
    
    
if __name__ == "__main__":
    client = CosUploader()
    os.system('cls')
    
    print("\n\n{:=^80}".format("腾讯云图床管理工具 V1.1"))

    print("\n 输入1上传文件")
    print("\n 输入2将本地的删除同步到 web 端\n")

    choice = input()
    if choice == "1":
        client.up() # 将 地点1 的所有文件，重命名到地点2，并上传到 web
    elif choice == "2":
        client.sync_local_to_web() #用于本地删除图片后，同步到 web 端删除
    else:
        print("错误的输入")
```

## 自动化

新建 `bat` 文件：

```cmd
cd C:\ProgramData\anaconda3\condabin
call activate
python C:\Users\Administrator\Pictures\CosUploader.py
pause
```

?> 激活虚拟环境时，要加个 call，不然会秒退; <br> call 命令可以在批处理程序中调用另一个批处理程序。

?> bat 脚本最后一行加上 pause，不会运行完就直接退出了

这样，运行这个 bat 脚本就会实现 图片上传及管理。

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220420114445.png">


改进思路：
- 给图片按日期分个文件夹
- 通过注册表为 windows 右键添加选项（上传），并自动将地址弄到剪贴板


参考资料：
- [腾讯云 COS SDK](https://cloud.tencent.com/document/product/436/12269)