推荐UP 主：
- [M_Studio](https://space.bilibili.com/370283072)
- [OneCredit](https://space.bilibili.com/504686800)


</br>

## _Day02_

[做一个 2D 小狐狸游戏](https://www.bilibili.com/video/BV1W4411Z7xs)

- 导入资源、Tile 地图（利用一个个小方块堆起游戏的场景）
- 图层（排序图层，图层顺序置1上层）
- 添加 Sprite，相当于一个组件，容器？
- 为其添加 物理系统、碰撞系统
- 使用 大的 VS2019
  - 查看代码辅助信息
  - 敲几个字母，一个 Tab，太舒服了
- Rigidbody2D
  - .velocity = new Vector2(move*speed,rb.velocity.y)
  - .transform.localScale = new Vector3(d, 1, 1)
- 解决帧数导致的移动差异
  - Update() -> FixedUpdate()
  - 运动坐标 * Time.deltaTime
- 添加动画
  - window， animation 为 AnimationController 新建一个动画
  - 在动画中插入图片（帧），调整合适的像素大小、速度
  - 在 Controller 中添加动画切换箭头，新增控制变量、动画切换条件
  - 代码中 Animator 示例.SetFloat("控制变量",Mathf.Abs()) 即可

- Update 中存放物理反馈有关的 ，btn_down；
  - 确保手感得到反馈，设置 press 值为 true
- FixedUpdate 中存放运动有关的；默认 0.02s 执行一次
  - 确保不同设备效果一致，设置 press false

cinemachine 
- follow 角色移动
- 调整 dead zone width，镜头不移动区域
- 添加 confider，使镜头不会超出 Polygon Collider 范围，istrigger

`day3`：

odin 插件，实现数据编辑器窗口。

https://www.bilibili.com/video/BV1Py4y1q7db


`day5`：

[3D RPG的人物移动](https://www.bilibili.com/video/BV13v411i76p)