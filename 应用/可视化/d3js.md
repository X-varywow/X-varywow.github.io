
> D3.js 是一个可以基于数据来操作文档的 JavaScript 库。


参考资料：
- [bili-数据可视化编程-使用D3.js](https://www.bilibili.com/video/av497590991)
- [课程 github](https://github.com/Shao-Kui/D3.js-Demos)
- [官网](https://d3js.org/) 
- [中文官网](https://d3js.org.cn/)

## 1. HelloWorld

vscode 插件 live server 运行代码即可；

画圆代码：
```html
<svg width="960" height="500" id="mainsvg" class="svgs"></svg>
<script>
    let mainsvg = d3.select('.svgs');

    let maingroup = mainsvg
        .append('g')
        .attr('transform', `translate(${100}, ${100})`);

    let circle = maingroup
        .append('circle')
        .attr('stroke', 'black')
        .attr('r', '66')
        .attr('fill', 'yellow');
</script>
```

## 2. 操控SVG 

### 2.1 选取

```js
d3.select("#id_name") // .class_name 也行
d3.selectAll(".class_name")

// 基于层级的查询
d3.select("#id_name rect")
```

### 2.2 增删

```js
//普通
const myRect = svg.append('rect')

// 链式
const myRect = d3.select('#mainsvg').append('rect').attr('x','100')

//普通
element.remove()
```

### 2.3 SVG

`SVG`：可缩放矢量模型；是 D3.js 主要操作的对象

常见的属性：
- id class
- x, y（屏幕坐标系）
- cx, cy（圆心的坐标）
- fill（填充颜色）
- stroke（边框的颜色）
- height, width, r（半径）
- transform -> translate（平移）, rotate（旋转）, scale（缩放）

```js
// 利用 attr 设置属性
d3.select("#id_name").attr("fill", "green")

d3.select("#id_name").attr("transform", "translate(100,100)")
```

```js
// 模板字符串的使用
// 模板字符串使用 `` 表示
let width =666;

.attr('transform', `translate(0,${width+100})`)
```

```js
//字符串转数值：

let val = +('3.14')
```

### 2.4 比例尺

```js
// 比例尺：用于把实际数据空间映射到屏幕空间
const myScale = d3.scaleLinear().domain([0,10]).range([-1000, 1000])

myScale(5) // out: 0

// 常结合读取的数据与 d3.max 等接口使用
const xScale = d3.scaleLinear()
    .domain([0, d3.max(data, d => d.value)])
    .range([0, innerWidth])

//Linear 定义域和值域都是连续的
//Band 定义域是离散的，值域是连续的
```

### 2.5 坐标轴

一个坐标轴为一个 group(\<g\>)

```js
const g = svg.append('g').attr('id', 'maingroup')
.attr('transform':'translate(${margin.left},${margin.top})')

const yAxis = d3.axisLeft(yScale);
g.append('g').call(yAxis);

const xAsix = d3.axisBottom(xScale);
g.append('g').call(xAxis).attr('transform', 'translate:(0, ${innerHeight})')
```


### 2.6 范例

```js
const data=[{},{}...]

const svg = d3.select("mainsvg");
const width = +svg.attr("width");
const height = +svg.attr("height");

const margin = {top:60, right:60, bottom:60, left:60} //防止出屏幕
const innerWidth = width - margin.left - margin.right
const innerHeight = height - margin.top - margin.bottom

const xScale = d3.scaleLinear()
    .domain([0, d3.max(data, d => d.value)])
    .range([0, innerWidth])

const yScale = d3.scaleBand()
    .domain( data.map(d => d.name))
    .range([0, innerHeight])
    .padding(0.1);

const g = svg.append('g').attr('id', 'maingroup')
.attr('transform':'translate(${margin.left},${margin.top})')

const yAxis = d3.axisLeft(yScale);
g.append('g').call(yAxis);

const xAsix = d3.axisBottom(xScale);
g.append('g').call(xAxis).attr('transform', 'translate:(0, ${innerHeight})')

// 根据 data 画一个柱状图
data.forEach( d =>{
    g.append('rect')
    .attr('width', xScale(d.value))
    .attr('height', yScale.bandwidth())
    .attr('fill', 'green')
    .attr('y', yScale(d.name))
    .attr('opacity', 0.8)
});

//更改坐标轴文字大小
d3.selectAll('.tick text').attr('font-size', '2em')

g.append("text").text('Members of CSCG')
.attr('font-size','3em')
.attr('transform', 'translate(${innerWidth/2}, 0)')
.attr('text-anchor', 'middle') //锚点居中
```

## 3. DataJoin

### 3.1 数据绑定

将 **数据与图元进行绑定**；相当于：对数据的一种抽象，使网页不再局限于某一具体的数据。

范例：
```js
d3.selaceAll('rect').data(dataArray)
    .attr('width', d => xScale(d.value)) //第二个参数，相当于 python 的 lambda
```

>默认的绑定按照索引顺序，
当数据更新后排布发生变化时，需使用 key

```js
// 将每个数据项的 name 作为 key
selection.data(dataArray, d => d.name)
```

### 3.2 动画

利用 transition()，duration()

```js
d3.selaceAll('rect').data(dataArray, d => d.name)
    .transition()
    .duration(1000)
    .attr('width', d => xScale(d.value)) //第二个参数，相当于 python 的 lambda
```

当情况复杂时，如动画中涉及到数据项的退出与引入，

有以下两种解决方式：
- `enter update exit`
  - .data(dataArray).enter().append().attr
  - update 就是 3.1 的内容
  - .data(dataArray).exit().remove()
    - 这里可以做个颜色隐去的动画
- `join`（简便，不灵活）

```js
// 图元：三角形、圆之类
.data(myData).join('图元').attr().attr()...
```
### 3.3 CSV 数据

```js
// .csv 返回一个 Promise 对象，用于执行异步操作
d3.csv('filename.csv').then(data => {
    // 数据处理
    // data = data.filter(d => d['地区'] !== '总计' );
    // data.forEach( d=>{
    //     d['确诊人数'] = +(d['确诊人数']);
    // });
    // console.log(data)
})
```

### 3.4 实例分析

!>先学些框架，实际应用时再看例子仔细学

例子：疫情确认人数变化的散点图（对应课程P5）

```html
<!-- scatter.html -->
<!DOCTYPE html>
<html>
  <head>
    <title>Scatter-Animation</title>
    <script src="d3.min.js"></script>
  </head>
  <body style="text-align: center">
    <svg width="1650" height="920" id="mainsvg" class="svgs" style="background-color: #ffffff;display: block; margin: 0 auto;"></svg>
    <script>
// d3.js 脚本
    </script>
  </body>
</html>
```

```js
// 1. 定义常量、变量
// width，margin, 匿名函数等
const svg = d3.select('#mainsvg');

// 2. renderinit
const renderinit = function(data, seq){
    // 定义坐标尺 xScale
    // 定义坐标 xAxis
    // 设置坐标文本 xAxisGroup
    // 设置 legend
}

// 3. renderupdate
const renderupdate = async function(seq){
    // 引入数据和动画
}

// 4. 主程
d3.csv('hubei_day14.csv').then(async function(data){
    // 数据处理

    renderinit(data, sequential[0]);

    for(let i = 0; i < sequential.length; i++){
        await renderupdate(sequential[i]);
    }
}
```

## 4. Path

### 4.1 图元 Path

`<Path>` 相当于一只画笔，如 Python 中的 turtle

- fill 填充
- stroke 描边
- transform
- **d 属性**
  - `M 10 10` # 画笔移动到 (10, 10)
  - `L 10 10` # 画笔勾勒到 (10, 10)
  - `H 90` # 画笔水平勾勒 90
  - `V 90` # 画笔竖直勾勒 90
  - `C x1 y1, x2 y2, x y` # 三阶贝塞尔曲线，[原理](https://www.jianshu.com/p/8f82db9556d2)
    - (x, y) 曲线的终点
    - (x1, y1) 起点的控制点
    - (x2, y2) 终点的控制点
  - S x y, endx endy # 平滑曲率
  - Q x y, endx endy # 二次贝塞尔曲线
  - T # 映射
  - A # 弧线
  - Z # 关闭路径

### 4.2 Path 生成器

`d3.line()`

可使用 curve 对点进行平滑拟合 [D3 curve explorer](http://bl.ocks.org/d3indepth/b6d4845973089bc1012dec1674d3aff8)

```js
const myArray = [
    {'x': 100, 'y': 100},{'x': 200, 'y': 300},
    {'x': 300, 'y': 50},{'x': 400, 'y': 600}
];
const pathLine = d3.line().x(d => d.x).y(d => d.y)
// .curve(d3.curveCardinal.tension(0.5)); // the line is guaranteed to pass through points; 
// .curve(d3.curveBasis); // the line is NOT guaranteed to pass through points; 
// .curve(d3.curveStep);  // 只走水平或竖直的线，阶梯状
//.curve(d3.curveNatural); 
// .curve(d3.curveLinear); // default
svg.append('path').attr('stroke', 'black').attr('fill', 'none')
.attr('d', pathLine(myArray));
```

`d3.arc()`


画出一个绿色的圆环片：
```js
const part1 = {'startAngle': 0, 'endAngle': 1.1855848768577961};
const pathArc1 = d3.arc().innerRadius(100).outerRadius(200);
svg.append('path').attr('stroke', 'black').attr('fill', 'green').attr('transform', 'translate(800, 400)')
.attr('d', pathArc1(part1));
```

[Bilibili-绘制动态折线图](https://www.bilibili.com/video/BV1HK411L72d?p=7)


## 5. Interaction

```html
<script src="topojson.js"></script>
<script src="d3-tip.js"></script>
```

### 5.1 地图数据

地图数据的表达：TopoJson & GeoJson

数据的读取：
```js
d3.json("file_name").then( data => {
    console.log(data);
    // geojson 数据转换成 topojson
    let worldmeta = topojson.feature(data, data.objects.countries);
});
```


代码实例：

```js
// 定义一个投影函数
const projection = d3.geoNaturalEarth1();

const geo = d3.geoPath().projection(projection)

d3.json("file_name").then( data => {
    let worldmeta = topojson.feature(data, data.objects.countries);

    projection.fitSize([innerWidth, innerHeight], worldmeta)

    g.selectAll('path').data(worldmeta.features).join('path')
    .attr('d', geo)
    .attr('stroke', 'black')
    .attr('stroke-width', 1)
});
```

### 5.2 事件机制

使用方法：`图元.on(事件类型，触发动作)`

- DOM Events
  - click
  - mouseover
  - mouseout
  - keydown
  - contextmenu 鼠标右键


tip 的使用：
```js
const tip = d3.tip()
.attr('class', 'd3-tip').html( d => d.properties.name);
svg.call(tip);

图元.on('click', function(d){
    tip.show(d);
})
```

还可以基于 d3-contour 和 d3-geo 绘制等值线图，参照 PPT5

## 6. Stack

`d3.stack()`，是 D3 提供的数据处理接口

当需要堆叠数据的时候，如：堆叠柱状图、主题河流时，可以使用；

## 7. tree & graph

d3.hierarchy 层级数据的处理与预计算

d3.tree & d3.partition 层级数据的划分与映射（树状图）（冰锥图、日晕图）

## 8. ToBeContinued

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220513001658.png">

>小结：这些还是需要熟练度的；作为需求较少的一项技能，需要时，去官网找些示例，改改代码

[D3-Gallery](https://observablehq.com/@d3/gallery)
