

</br>

## _文件说明_

`.h`（头文件）和 `.cpp`（源文件）通常用于实现代码的模块化和封装，让代码更易维护和复用。


```cpp
// myclass.h
#ifndef MYCLASS_H  // 避免重复包含
#define MYCLASS_H

class MyClass {
private:
    int value;
public:
    MyClass(int val);  // 构造函数声明
    void showValue();  // 成员函数声明
};

#endif  // 结束预处理指令
```


```cpp
// myclass.cpp
#include <iostream>
#include "myclass.h"

using namespace std;

MyClass::MyClass(int val) : value(val) {}  // 构造函数定义

void MyClass::showValue() {  // 函数定义
    cout << "Value: " << value << endl;
}

```

.h 只做声明，不做实现

.cpp 文件主要用于定义在 .h 头文件中声明的函数和类，并包含实现细节。


</br>

## _数据结构_


</br>

## _逻辑片段_


</br>

## _other_

```cpp
// 允许直接使用标准库的双向链表容器 list ，等价于 std::list
using std::list;
```