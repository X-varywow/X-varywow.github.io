

它是 CSS 的超集，在传统的 CSS 上引入了变量等概念。

```html
<style lang="scss">
@use '@/assets/css/normalize.css';
</style>
```

SCSS中的变量以 $ 开头

```css
$border-color:#aaa; //声明变量
.container {
    $border-width:1px;
    border:$border-width solid $border-color; //使用变量
}
```