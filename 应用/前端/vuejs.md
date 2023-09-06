
Vue 是一套用于构建用户界面的渐进式框架。

参考资料：
- [官方文档](https://cn.vuejs.org/guide/introduction.html)

### 1. 引入
```html
<!-- 开发环境版本，包含了有帮助的命令行警告 -->
<script src="https://cdn.jsdelivr.net/npm/vue@2/dist/vue.js"></script>
```

```html
<!-- 生产环境版本，优化了尺寸和速度 -->
<script src="https://cdn.jsdelivr.net/npm/vue@2"></script>
```

-------------------------

`2022.12`（上面的方式是一年前的）

```bash
#创建一个 vue 应用
npm init vue@latest
```

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20221228020404.png">


```bash
#将应用发布到生产环境
npm run build
```

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20221228022153.png">

### 2. 核心

Vue.js 的核心是一个允许采用简洁的模板语法来声明式地将数据渲染进 DOM 的系统：

```html
<div id="app">
  {{ message }}
</div>
```

```js
var app = new Vue({
  el: '#app',
  data: {
    message: 'Hello Vue!'
  }
})
```

### 3. 基础

每个 Vue 应用都是通过 createApp 函数创建一个新的 应用实例：

```js
import { createApp } from 'vue'

const app = createApp({
  /* 根组件选项 */
})
```

一个待办事项的组件：

```
App (root component)
├─ TodoList
│  └─ TodoItem
│     ├─ TodoDeleteButton
│     └─ TodoEditButton
└─ TodoFooter
   ├─ TodoClearButton
   └─ TodoStatistics
```


