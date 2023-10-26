
```python
%matplotlib inline
#import pandas as pd
#from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10,10)
##### 绘制sigmoid图像
fig = plt.figure()
y_sigmoid = 1/(1+np.exp(-x))
ax = fig.add_subplot(321)
ax.plot(x,y_sigmoid,color='blue')
ax.grid()
ax.set_title('(a) Sigmoid')
ax.spines['right'].set_color('none') # 去除右边界线
ax.spines['top'].set_color('none') # 去除上边界线
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

##### 绘制Tanh图像
ax = fig.add_subplot(322)
y_tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
ax.plot(x,y_tanh,color='blue')
ax.grid()
ax.set_title('(b) Tanh')
ax.spines['right'].set_color('none') # 去除右边界线
ax.spines['top'].set_color('none') # 去除上边界线
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

##### 绘制Relu图像
ax = fig.add_subplot(323)
y_relu = np.array([0*item  if item<0 else item for item in x ])
ax.plot(x,y_relu,color='darkviolet')
ax.grid()
ax.set_title('(c) ReLu')
ax.spines['right'].set_color('none') # 去除右边界线
ax.spines['top'].set_color('none') # 去除上边界线
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

##### 绘制Leaky Relu图像
ax = fig.add_subplot(324)
y_relu = np.array([0.2*item  if item<0 else item for item in x ])
ax.plot(x,y_relu,color='darkviolet')
ax.grid()
ax.set_title('(d) Leaky Relu')
ax.spines['right'].set_color('none') # 去除右边界线
ax.spines['top'].set_color('none') # 去除上边界线
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

##### 绘制ELU图像
ax = fig.add_subplot(325)
y_elu = np.array([2.0*(np.exp(item)-1)  if item<0 else item for item in x ])
ax.plot(x,y_elu,color='darkviolet')
ax.grid()
ax.set_title('(d) ELU alpha=2.0')
ax.spines['right'].set_color('none') # 去除右边界线
ax.spines['top'].set_color('none') # 去除上边界线
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

ax = fig.add_subplot(326)
y_sigmoid_dev = y_sigmoid*(1-y_sigmoid)
ax.plot(x,y_sigmoid_dev,color='green')
ax.grid()
ax.set_title('(e) Sigmoid Dev')
ax.spines['right'].set_color('none') # 去除右边界线
ax.spines['top'].set_color('none') # 去除上边界线
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

plt.tight_layout()
plt.savefig('Activation.png')
plt.show()
```