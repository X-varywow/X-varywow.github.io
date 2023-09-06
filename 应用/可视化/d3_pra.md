> 大作业笔记

- [d3-demo](https://github.com/zjw666/D3_demo)
- https://d3-graph-gallery.com/ ⭐



### 1. 简要原理

```html
<!-- 先在 body 中建立一个 svg 画布 -->
<svg width="960" height="500" id="mainsvg" class="svgs"></svg>

<!-- 之后使用 select append -->
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

### 2. CSV 数据读入并处理

```js
// 读入 csv 并过滤，打印
d3.csv("/data/hubei_day14.csv").then(data => {
    data = data.filter(d => d['地区'] !== '总计')

    
    console.log(data)
})
```

### 3. 主题河流代码分析

>参考 [示例](https://d3-graph-gallery.com/streamgraph.html)，一部分一部分删除代码，留下核心，之后一部分一部分改动；我竟然是这样完成作业的。。。

主题河流使用到了 d3.area() 和 堆叠数据

#### 3.1 框架

```js
// set the dimensions and margins of the graph
const margin = { top: 0, right: 30, bottom: 50, left: 50 },
    width = 1200 - margin.left - margin.right,
    height = 800 - margin.top - margin.bottom;

// append the svg object to the body of the page
const svg = d3.select("#mainsvg")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
        `translate(${margin.left}, ${margin.top})`);
```

之后读入 csv 并在该函数中 新添图元即可

#### 3.2 核心逻辑

```js
// Parse the Data
d3.csv("/data/yixiang_log.csv").then(function (data) {

    // List of groups = header of the csv files
    const keys = data.columns.slice(1)

    // Add X axis
    const x = d3.scaleBand()
        .domain(data.map(d => d.year))
        .range([0, width]);
    svg.append("g")
        .attr("transform", `translate(-50, ${height * 0.8})`)
        .style("font-size", 17)
        .call(d3.axisBottom(x))

    // Add X axis label
    svg.append("text")
        .attr("text-anchor", "end")
        .style("font-size", 22)
        .attr("x", width)
        .attr("y", height - 30)
        .text("朝代");

    // Add Y axis
    const y = d3.scaleLinear()
        .domain([-60, 60])
        .range([height, 0]);

    // color
    const color = d3.scaleOrdinal()
        .domain(keys)
        .range(d3.schemeTableau10);

    //stack the data
    const stackedData = d3.stack()
        .offset(d3.stackOffsetSilhouette)
        .keys(keys)
        (data)

    // Area generator
    const area = d3.area()
        .x(function (d) { return x(d.data.year); })
        .y0(function (d) { return y(d[0]); })
        .y1(function (d) { return y(d[1]); })


    // Show the areas
    svg
        .selectAll("mylayers")
        .data(stackedData)
        .join("path")
        .attr("class", "myArea")
        .style("fill", function (d) { return color(d.key); })
        .attr("d", area)
    })
```

#### 3.3 添加 tooltip 交互（3.2 代码块中）

```js
// create a tooltip
const Tooltip = svg
    .append("text")
    .attr("x", -50)
    .attr("y", 100)
    .style("opacity", 0)
    .style("font-size", 22)

// Three function that change the tooltip when user hover / move / leave a cell
const mouseover = function (event, d) {
    Tooltip.style("opacity", 1)
    d3.selectAll(".myArea").style("opacity", .2)
    d3.select(this)
        .style("stroke", "white")
        .style("opacity", 1)
}
const mousemove = function (event, d, i) {
    grp = d.key
    Tooltip.text("意象：" + grp)
}
const mouseleave = function (event, d) {
    Tooltip.style("opacity", 0)
    d3.selectAll(".myArea").style("opacity", 1).style("stroke", "none")
}

// 并在原本代码的 svg 中添加事件监听函数
// .on("mouseover", mouseover)
// .on("mousemove", mousemove)
// .on("mouseleave", mouseleave)
```

#### 3.4 添加 legend 标签（3.2 代码块中）

```js
//show the legend

var side = d3.select("#my_dataviz2")
// Add one dot in the legend for each name.
side.selectAll("mydots")
    .data(keys)
    .enter()
    .append("circle")
    .attr("cx", 100)
    .attr("cy", function (d, i) { return 100 + i * 25 }) // 100 is where the first dot appears. 25 is the distance between dots
    .attr("r", 8)
    .style("fill", function (d) { return color(d) })

// Add one dot in the legend for each name.
side.selectAll("mylabels")
    .data(keys)
    .enter()
    .append("text")
    .attr("x", 120)
    .attr("y", function (d, i) { return 100 + i * 25 }) // 100 is where the first dot appears. 25 is the distance between dots
    .style("fill", function (d) { return color(d) })
    .text(function (d) { return d })
    .attr("text-anchor", "left")
    .style("alignment-baseline", "middle")
    .style("font-size", 22)
```

### 4. 欣赏一下

（首页）
<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220609012752.png">


（河流图）
<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220609012749.png">


（关系图）
<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220609012750.png">