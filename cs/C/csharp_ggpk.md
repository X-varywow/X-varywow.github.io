

核心目的是想研究一下 poe2 中的补丁，用以达到：全图、去迷雾等

##  OO2Core

游戏里非常常见的一个 Oodle 解压缩运行库 DLL，很多现代游戏（UE4/UE5、Path of Exile、Warframe、Fortnite 等）都会用它来压缩资源包。

游戏资源（贴图、mesh、shader、音频等）通常不会裸存， 由于其压缩率高、解压快，所以选用了 `oo2core`

---------

对于 poe2 的 content.ggpk, oo2core.dll 是解压逆向过程的一个**核心依赖**，

同时 oodle 是商业闭源 SDK, 通常需要去游戏目录或网上下载。

poe2 引擎是 GGG 自研的 C++ ARPG 引擎，由于地图资源极多，ggpk 系统在其中比较核心。

---------

类比一下，其它游戏也会有资源包文件，不可能几万个小文件直接散落；

如 UE 的 `.pak`, unity 的 `assets` ... 

感觉 GGG 自己于 2010 做的一套这一套打包体系比较 low


## 逆向过程

拿到 oo2core 之后，才刚刚开发。。。 很复杂

- 索引系统 (资源路径或 hash 到 chunk/offset 的查找表)
- bundle 映射 (把“逻辑资源名”映射到实际 bundle 文件和压缩块的位置)
- 自定义格式
- schema ( `.dat` 二进制列数据，转成可理解的)
- shader
- mesh pipeline


这边直接使用 libggpk3 了，不做研究

https://github.com/aianlinb/LibGGPK3


使用仓库内部的 demo 结合 ai 基本可以实现想要的功能


## patch1. minimap visibility

`.hlsl` 是微软的 GPU 着色器编程语言，由 GPU 执行，用来控制光照、材质、特效等渲染效果。

语法上接近 c 语言，运转过程： HLSL源码 -> fxc/dxc 编译 -> DXBC/DXIL -> GPU 执行



demo:

```c
// 输出红色
float4 main() : SV_Target
{
    // (r, g, b, alpha透明度)
    return float4(1,0,0,1);
}

// SV_Target：返回值是最终输出到渲染目标(Render Target)的颜色
// SV = System Value
```

ai修改这两个文件：
shaders/minimap_visibility_pixel.hlsl
shaders/minimap_blending_pixel.hlsl



## patch2. sight distance

metadata/characters/character.ot

```c
// 新增了 on_initial_position_set 那两行
// 0.4 行， 0.5 记得在报错

Positioned 
{
	on_initial_position_set = "ClearCameraZoomNodes();"
	on_initial_position_set = "CreateCameraZoomNode(3000.0, 3000.0, 1.6);"
	object_size = 0
	secondary_object_size = 4
	object_targeting_size = 3
	base_pushiness = 30
	blocking = true
	team = 1
}
```



## patch3. worldmap fog

metadata/materials/environment/worldmap/worldmap_fogowar.fxgrapg

将所有内容去掉，改成如下

```c
{
    "version": 6,
}
```