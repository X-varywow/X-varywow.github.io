
_shutil_

提供了一系列对文件和文件集合的高阶操作。 文件拷贝、删除等。 

官方文档：https://docs.python.org/zh-cn/3/library/shutil.html

</br>

（1）使用 make_archive 打包文件

```python
import shutil
import os
from google.colab import files

def zip_directory(directory_path, zip_path):
    shutil.make_archive(zip_path, 'zip', directory_path)

# Set the directory path you want to download
directory_path = '/content/roop/video3'

# Set the zip file name
zip_filename = 'video3.zip'

# Zip the directory
zip_directory(directory_path, zip_filename)

# Download the zip file
files.download(zip_filename+'.zip')
```