
`NumPy` is the fundamental package for **scientific computing** in Python.

------

```bash
pip install numpy
```

</br>

## _Numpy 数组_

| 属性        | 说明                       |
| ----------- | -------------------------- |
| `.ndim`     | 秩，即轴的数量或维度的数量 |
| `.shape`    | n行m列                     |
| `.size`     | n*m个数                    |
| `.dtype`    | 元素类型                   |
| `.itemsize` | 元素大小，单位：字节       |

ndarray数组可以由非同质对象构成。
非同质ndarray元素为对象类型，`dtype('O')`

### （1）创建方法
1. `np.array(list/tuple,dtype=np.float32)`

```python
# 一维数组
np.array([1,2,3])

# 二维数组
np.array([(1,2,3),(4,5,6)])

# 等差数组
np.arange(start,stop,step)
```

2. `np.arange(n)`
3. `np.ones(shape)`
4. `np.zeros(shape)`
5. `np.full(shape,val)`
6. `np.eye(n)`
7. `np.ones_like(a)`
8. `np.zeros_like(a)`
9.  `np.full_like(a,val)`
10. `np.linspace(n,m,k,endpoint=True)`




11. `np.concatenate()` 

12. `np.random.uniform()`

```python
np.random.uniform(low = 0.0, high = 1.0, size = 4)
```


### （2）变换方法
1. `a.reshape(shape)`，对a`reshape`,返回shape形状数组
2. `a.resize(shape)`，同`reshape`，但修改原数组
3. `a.swapaxes(ax1,ax2)`，调换维度
4. `a.flatten()`，降维，不改变a
5. `new_a=a.astype(new_type)`
6. `a.tolist()`


-----------------

数组叠加：

```python
# 在 数组 a 末尾水平追加数组 b
np.hstack(a, b)

# 数组堆叠到 a 上
np.vstack(a, b)
```









### （3）切片和子集
1. 索引切片同 `list`，但：多维间用 `,` 隔开，eg: `array[i,j]`
2. `:` 可选取整个维度
3. 还有一种布尔索引，eg: `array[i<4]`

### （4）运算方法
1. `a.mean()`
2. 元素群运算：`+` `-` `*` `/` `**`
  `np.maxmum(x,y)`
  `np.minimum(x,y)`
  算术比较，返回bool值
3. `np.abs()`，`np.fabs()`
4. `np.sqrt()`，`np.square()`
5. `np.log()`，`np.log2()`
6. `np.ceil()`，`np.floor()` 
7. `np.rint()`，四舍五入
8. `np.sin()`，`np.cos()`
9. `np.exp()`，计算指数
10. `np.sign()`，返回1，0，-1 


**实践**
```python
import numpy as np
np.arange(5)            #-->array([0, 1, 2, 3, 4])
np.ones([2,2])          #-->array([[1., 1.],1., 1.]])
np.linspace(1,10,4)     #-->array([ 1.,  4.,  7., 10.])
```

</br>

## _统计函数_


1. `np.sum(a,axis=None)`
2. `np.mean(a,axis=None)`，计算期望
3. `np.average(a,axis=None,weights=None)`，计算加权平均
4. `np.std(a,axis=None)`，计算标准差
5. `np.var(a,axis=None)`，计算方差

此外，数组本身的方法也适用

6. `array.sum()`
7. `array.min()`
8. `array.max()`
9. `array.cumsum()`，指定轴求累计和


</br>

## _特殊函数_

`np.gradient(f)` 计算斜率


`np.random.uniform(low,high,size)` 产生均匀分布数组


`np.random.normal(loc,scale,size)` 产生正态分布数组，`loc`均值，`scale`标准差

`np.random.poisson(lam,size)` 产生泊松分布数组，`lam`随机事件发生概率


---------------------

文件读写

**一维或二维**
1. `np.savetxt(frame,array,fmt,delimiter=None)`
其中delimiter应该为 `,`
eg. `np.savetxt('foo.csv',a,fmt='%d',delimiter=',')`
1. `np.loadtxt(frame,dtype=np.float,delimiter=None,unpack=False)`

**多维**
1. ` a.tofile()`
2. `np.fromfile`

```python
import numpy as np 
 
b = np.load('outfile.npy')  
print (b)
```


```python
import numpy as np 
 
a = np.array([1,2,3,4,5]) 
 
# 保存到 outfile.npy 文件上
np.save('outfile.npy',a) 
 
# 保存到 outfile2.npy 文件上，如果文件路径末尾没有扩展名 .npy，该扩展名会被自动加上
np.save('outfile2',a)
```

</br>

## _other_

#### 1. 转置矩阵

```python
#做法一：
import numpy as np
class Solution:
    def transpose(self, m: List[List[int]]) -> List[List[int]]:
        return np.matrix(m).T.tolist()

#做法二：
class Solution:
    def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
        return list(zip(*matrix))
```

#### 2. 对图片进行手绘风格转变

```python
from PIL import Image
import numpy as np
 
a = np.asarray(Image.open('./beijing.jpg').convert('L')).astype('float')
 
depth = 10.                      # (0-100)
grad = np.gradient(a)             #取图像灰度的梯度值
grad_x, grad_y = grad               #分别取横纵图像梯度值
grad_x = grad_x*depth/100.
grad_y = grad_y*depth/100.
A = np.sqrt(grad_x**2 + grad_y**2 + 1.)
uni_x = grad_x/A
uni_y = grad_y/A
uni_z = 1./A
 
vec_el = np.pi/2.2                   # 光源的俯视角度，弧度值
vec_az = np.pi/4.                    # 光源的方位角度，弧度值
dx = np.cos(vec_el)*np.cos(vec_az)   #光源对x 轴的影响
dy = np.cos(vec_el)*np.sin(vec_az)   #光源对y 轴的影响
dz = np.sin(vec_el)              #光源对z 轴的影响
 
b = 255*(dx*uni_x + dy*uni_y + dz*uni_z)     #光源归一化
b = b.clip(0,255)
 
im = Image.fromarray(b.astype('uint8'))  #重构图像
im.save('./beijingHD.jpg')
```

#### 3. 使用 numpy 加速计算

```python
import numpy as np

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

arr = np.random.randint(1, 100, size = 100000)

def way1():
    tot = 0
    cnt = 0
    for num in arr:
        if num<70:
            tot += num
            cnt += 1
    return tot/cnt

def way2():
    res = 0
    res = arr[arr<70].mean()
    return res

%timeit way1()

%timeit way2()

# -> 14.5 ms ± 36.6 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# -> 717 µs ± 7.15 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# 性能是其 20 倍
```



-----------


参考资料：
- [Numpy官网 初学者文档](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [Numpy官网 文档](https://numpy.org/doc/stable/reference/)
- [菜鸟教程 - Numpy](https://www.runoob.com/numpy/numpy-tutorial.html)
