
## 1. 入门

参考资料：https://phaser.io/tutorials/making-your-first-phaser-3-game-chinese

### 1.1 环境准备

引入 js 库，用服务器方式运行网页即可。（LiverServer）

```html
<script src="//cdn.jsdelivr.net/npm/phaser@3.55.2/dist/phaser.min.js"></script>
```

### 1.2 代码结构

```js
var config = {
    type: Phaser.AUTO,  // Phaser.CANVAS 或者 Phaser.WEBGL，优先 WebGL
    width: 800,
    height: 600,
    scene: {
        preload: preload,
        create: create,
        update: update
    }
};

// 将 Phaser.Game对象实例 赋值给 game 局部变量
var game = new Phaser.Game(config);

function preload ()
{
}

function create ()
{
}

function update ()
{
}
```

### 1.3 加载资源

```js
// key - val 的方式定义好资源
function preload ()
{
    this.load.image('sky', 'assets/sky.png');
    this.load.image('ground', 'assets/platform.png');
    this.load.image('star', 'assets/star.png');
    this.load.image('bomb', 'assets/bomb.png');
    this.load.spritesheet('dude', 
        'assets/dude.png',
        { frameWidth: 32, frameHeight: 48 }
    );
}

// x, y 坐标
// 游戏对象的定位都默认基于中心点。
// 游戏对象的显示顺序与你生成它们的顺序一致，即图层优先级
function create ()
{
    this.add.image(400, 300, 'sky');
    this.add.image(400, 300, 'star');
}
```



```js
// 使用列表 创建资源
var platforms;

function create ()
{

    platforms = this.physics.add.staticGroup();

    platforms.create(400, 568, 'ground').setScale(2).refreshBody();

    platforms.create(600, 400, 'ground');
    platforms.create(50, 250, 'ground');
    platforms.create(750, 220, 'ground');
}
```

### 1.4 玩家

```js
// 生成物理精灵
player = this.physics.add.sprite(100, 450, 'dude');

player.setBounce(0.2);
player.setCollideWorldBounds(true);

// 动画
this.anims.create({
    key: 'left',
    frames: this.anims.generateFrameNumbers('dude', { start: 0, end: 3 }),
    frameRate: 10,
    repeat: -1
});

this.anims.create({
    key: 'turn',
    frames: [ { key: 'dude', frame: 4 } ],
    frameRate: 20
});

this.anims.create({
    key: 'right',
    frames: this.anims.generateFrameNumbers('dude', { start: 5, end: 8 }),
    frameRate: 10,
    repeat: -1
});
```

### 1.5 碰撞系统

```js
this.physics.add.collider(player, platforms);
```

## 2. 样例学习

参考：https://github.com/channingbreeze/games


http://labs.phaser.io/edit.html?src=src\game%20objects\shapes\polygon.js

```js
// 增加动画
    this.tweens.add({

        targets: r4,
        scaleX: 0.25,
        scaleY: 0.5,
        yoyo: true,
        repeat: -1,
        ease: 'Sine.easeInOut'

    });
```


## 3. 实战

#### 2.5 打包

使用 electron