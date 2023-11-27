
Vue 是一套用于构建用户界面的 JavaScript 框架。提供了模版语法帮助快速开发界面。

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
npm create vue@latest
```

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20221228020404.png">


```bash
cd project_path
npm install
npm run dev


# 发布到生产环境时, ./dist 下会创建一个生产环境的构建版本
npm run build
```

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20221228022153.png">

### 2. 核心

Vue.js 的核心是允许采用简洁的 **模板语法** 来 <u> 声明式地将数据渲染进 DOM</u>

- 声明式渲染，即模版语法
- 响应性，自动追踪 js 状态并在变化时响应地更新 DOM




Demo: hello world

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

Demo: 计数器

```html
<div id="app">
  <button @click="count++">
    Count is: {{ count }}
  </button>
</div>
```

```js
import { createApp, ref } from 'vue'

createApp({
  setup() {
    return {
      count: ref(0)
    }
  }
}).mount('#app')
```












### 3. 语法说明

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

### other

组合式 API 

```html
<script setup>
import { ref, onMounted } from 'vue'

// 响应式状态
const count = ref(0)

// 用来修改状态、触发更新的函数
function increment() {
  count.value++
}

// 生命周期钩子
onMounted(() => {
  console.log(`The initial count is ${count.value}.`)
})
</script>

<template>
  <button @click="increment">Count is: {{ count }}</button>
</template>
```







--------------

参考资料：
- [官方文档](https://cn.vuejs.org/guide/introduction.html)
