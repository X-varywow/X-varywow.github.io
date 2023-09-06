

### 综合实验

![](https://img-1301102143.cos.ap-beijing.myqcloud.com/202112290017891.jpg)

**实验要求：**

?>1、 根据拓朴图分别在 S2628 和 S5750 创建相应 VLAN，并在 S2628 上将 F0/1-2 加入 VLAN2，将
F0/3-4 加入 VLAN4，在 S5750 上将 F0/1-2 加入 VLAN3。连接与 2628 和 5750 交换机相连的
测试机， 并进行基本参数的基本配置（20 分）

在 `S2628` 上：
```cmd
en
configure t
vlan 2
exit
vlan 4
exit

int f 0/1
switchport access vlan 2
exit
int f 0/2
switchport access vlan 2
exit

int f 0/3
switchport access vlan 4
exit
int f 0/4
switchport access vlan 4
end

show vlan
```

在 `S5750` 上：
```cmd
en
configure t
vlan 3
exit

int f 0/1
switchport access vlan 3
exit
int f 0/2
switchport access vlan 3
end

show vlan
```

并在图形化界面对测试机的 IP地址 进行配置，`24`代表子网掩码`255.255.255.0` 

?>2、 S5750 与 S2628 两台设备连接接口 F0/5、F0/6 作为 TRUNK 端口，建立 TRUNK 链路（10 分）。

在 `S2628` 上：
```cmd
en
configure t

int f 0/5
switchport mode trunk
exit

int f 0/6
switchport mode trunk
end

show int f 0/5 switchport
show int f 0/6 switchport
```

在 `S5750` 上：
```cmd
en
configure t

int f 0/5
switchport trunk encap dot1q //先封装协议再设置trunk端口
switchport mode trunk   //在三层交换机上直接使用该指令会报错
exit

int f 0/6
switchport trunk encap dot1q
switchport mode trunk
end

show int f 0/5 switchport
show int f 0/6 switchport
```

也可以使用 range 写法：
```cmd
int range f 0/5-6
```

?>3、 S5750 与 S2628 两台设备运行快速生成树协议（RSTP），并且要求 S2628 为根交换机。
或者将 F0/5、F0/6 连接的两条链路配置成聚合链路。（ 10 分）


在 `S2628` 上：
```cmd
configure t
spanning-tree mode rapid-pvst     //指定生成树协议的类型为RSTP 
spanning-tree vlan 2 priority 4096 //为 vlan 设置优先级
spanning-tree vlan 4 priority 4096 //4096小于默认的32768，会成为根交换机
end

show spanning-tree
```

在 `S5750` 上：
```cmd
configure t
spanning-tree mode rapid-pvst
end

show spanning-tree
```

?>4、 S5750 通过 SVI 方式和 RA 互连 （10 分）

SVI 指为交换机中的VLAN创建的虚拟接口，通过它可以实现 VLAN 间通信

```cmd
en
configure t

int f 0/7

int vlan 7
ip address 192.168.1.2 255.255.255.0
no shut
exit
show ip route
```

在 `RA` 上：
```cmd
en
configure t

int f 0/0
ip address 202.99.1.1 255.255.255.252
end
```
https://blog.51cto.com/microdq/1953462

?>5、 配置交换机连接测试机的网络参数，并在 S5750 上做相应配置，使得 VLAN 间可以互相访问，
所有地址配置正确（10 分）


?>6、 RA 和 RB 之间采用 PPP 链路，采用 PAP 或 CHAP 方式进行验证提高链路的安全性。（15 分）


?>7、 连接 RB 右侧的测试机并配置网络基本参数，在全网运应 RIPV2 或 OSPF 实现全网互连。（15
分）


?>8、 通过相关命令显示相关配置结果，并进行验证 （10 分）

Ping 不通啊，麻了。。。

`2021.12.29`，期末结束，应该多看看那两本《网络互联技术》


### other

成绩组成：
- 平时，50%
- 期末，50%

参考资料：
- 学校课程
- 《网络互联技术（理论篇）》
- 《网络互联技术（实践篇）》