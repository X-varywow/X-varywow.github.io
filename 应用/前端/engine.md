



## _pixi_

The HTML5 Creation Engine

Create beautiful digital content with the fastest, most flexible 2D WebGL renderer.

官网：https://pixijs.com/



```js
import * as PIXI from 'pixi.js';

const app = new PIXI.Application({
    background: '#1099bb',
    resizeTo: window,
});

document.body.appendChild(app.view);

// create a new Sprite from an image path
const bunny = PIXI.Sprite.from('https://pixijs.com/assets/bunny.png');

// center the sprite's anchor point
bunny.anchor.set(0.5);

// move the sprite to the center of the screen
bunny.x = app.screen.width / 2;
bunny.y = app.screen.height / 2;

app.stage.addChild(bunny);

// Listen for animate update
app.ticker.add((delta) =>
{
    // just for fun, let's rotate mr rabbit a little
    // delta is 1 if running at 100% performance
    // creates frame-independent transformation
    bunny.rotation += 0.1 * delta;
});

```


</br>

## _Three.js_

官网：https://threejs.org/



</br>

## _Orillusion_


[Orillusion - 次时代 WebGPU 引擎](https://www.orillusion.com/)



webgpu 和 webgl 是两种不同的 web图形 API， webgpu 可以更好地利用硬件加速；webgl 是 opengl es 的分支


现代三大图形API
- Direct3D (windows ps5)
- Metal(mac ios)
- Vulkan(android 跨平台)



--------

参考资料：
- [WebGPU 令人兴奋的 Web 发展](https://xie.infoq.cn/article/0c3d566af7edc86496d3565f5)


</br>

## _other_

web 图形：https://juejin.cn/post/7030464678400622600

