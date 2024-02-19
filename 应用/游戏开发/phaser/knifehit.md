参考资料：https://github.com/channingbreeze/games

体验地址：http://game.webxinxin.com/knifehit3/


## 代码说明

目录结构：
- assets
  - knife.png
  - target.png
- main.js
- phaser.min.js
- index.html


```html
<!-- index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>飞刀游戏</title>
    <script src="/phaser.min.js"></script>
    <style type="text/css">
        body{
            background: #000;
            padding: 0;
            margin: 0;
        }
        canvas{
            display: block;
            margin: 0;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

    </style>
</head>
<body>
    <script src="main.js"></script>
</body>
</html>
```

```js
// main.js
var game;
var scoreText

var gameOptions = {

    // 设置旋转速度，即每一帧转动的角度
    rotationSpeed: 1,
    // 刀飞出去的速度, 即一秒中内移动像素
    throwSpeed: 150,
    //v1.1新增 两把刀之前的最小角度(约束角度)
    minAngle: 10,

    //v1.2新增 最大转动的变化量，即每一帧上限
    rotationVariation: 1,
    //v1.2新增 下一秒的变化速度
    changeTime: 2000,
    //v1.2新增 最大旋转速度
    maxRotationSpeed: 2
}

// 按比例调整窗口
function resize() {
    var canvas = document.querySelector("canvas");
    var windowWidth = window.innerWidth;
    var windowHeight = window.innerHeight;
    var windowRatio = windowWidth / windowHeight;
    var gameRatio = game.config.width / game.config.height;
    if (windowRatio < gameRatio) {
        canvas.style.width = windowWidth + "px";
        canvas.style.height = (windowWidth / gameRatio) + "px";
    }
    else {
        canvas.style.width = (windowHeight * gameRatio) + "px";
        canvas.style.height = windowHeight + "px";
    }
}

// js 浏览器对象
window.onload = function () {
    var config = {
        type: Phaser.CANVAS,
        width: 720,
        height: 1280,
        backgroundColor: 0x444444,
        scene: [main_scene]
    };
    game = new Phaser.Game(config)
    window.focus() // 获得窗口焦点
    resize()       // 调整窗口
    window.addEventListener("resize", resize, false)
}

// 游戏主场景，继承
class main_scene extends Phaser.Scene {

    constructor() {
        super("main_scene")
    }

    // 资源预加载
    preload() {
        this.load.image("target", "assets/target.png")
        this.load.image("knife", "assets/knife.png")
    }

    // 游戏开始
    create() {
        this.currentRotationSpeed = gameOptions.rotationSpeed
        this.newRotationSpeed = gameOptions.rotationSpeed
        this.canThrow = true

        // 旋转的刀
        this.knifeGroup = this.add.group()
        // 精灵：刀、目标
        this.knife = this.add.sprite(game.config.width / 2, game.config.height / 5 * 4, "knife")
        this.target = this.add.sprite(game.config.width / 2, 400, "target")

        this.target.depth = 1

        // 计分板
        scoreText = this.add.text(16, 16, 'score: 0', { fontSize: '32px', fill: '#000' });

        // 点击后飞出刀
        this.input.on("pointerdown", this.throwKnife, this)

        // 创建循环的时间事件
        var timedEvent = this.time.addEvent({
            delay: gameOptions.changeTime,
            callback: this.changeSpeed,
            callbackScope: this,
            loop: true
        });
    }

    changeSpeed() {
        // 随机产生一个旋转方向
        var sign = Phaser.Math.Between(0, 1) == 0 ? -1 : 1;
        var variation = Phaser.Math.FloatBetween(-gameOptions.rotationVariation, gameOptions.rotationVariation);
        this.newRotationSpeed = (this.currentRotationSpeed + variation) * sign;
        this.newRotationSpeed = Phaser.Math.Clamp(this.newRotationSpeed, -gameOptions.maxRotationSpeed, gameOptions.maxRotationSpeed);
    }

    // 飞出刀动作
    throwKnife() {

        // 检查是否要飞出
        if (this.canThrow) {

            this.canThrow = false;

            // 飞刀的补间动画
            this.tweens.add({
                // 刀
                targets: [this.knife],
                // 到达的位置
                y: this.target.y + this.target.width / 2,

                // 补间速度
                duration: gameOptions.throwSpeed,

                // 回传范围
                callbackScope: this,

                // 执行后的回调函数
                onComplete: function (tween) {

                    //v1.1 新增 合法飞出参数
                    var legalHit = true;
                    //v1.1 新增 已经在圆木上刀成员
                    var children = this.knifeGroup.getChildren();
                    //v1.1 对于在圆木上的每一把刀设置约束角度
                    for (var i = 0; i < children.length; i++) {

                        //v1.1 判断当前飞刀与圆木上的刀是否在约束范围之内
                        if (Math.abs(Phaser.Math.Angle.ShortestBetween(this.target.angle, children[i].impactAngle)) < gameOptions.minAngle) {

                            //v1.1 确定标记参数
                            legalHit = false;

                            //v1.1 一旦在约束范围内就停止
                            break;
                        }
                    }
                    // 原来合法飞出
                    if (legalHit) {

                        // 玩家现可以再次扔刀
                        this.canThrow = true;

                        // 将飞出的刀插在圆木上
                        var knife = this.add.sprite(this.knife.x, this.knife.y, "knife");
                        //v1.1 飞刀的约束角度等于目标的角度
                        knife.impactAngle = this.target.angle;
                        // 将飞刀绑定在飞刀组中
                        this.knifeGroup.add(knife);

                        // 确定相对位置
                        this.knife.y = game.config.height / 5 * 4;

                        // 更新计分板
                        scoreText.setText('Score: ' + this.knifeGroup.getChildren().length);
                    }
                    else {
                        //v1.1 增加飞刀掉落动画
                        this.tweens.add({
                            targets: [this.knife],
                            y: game.config.height + this.knife.height,
                            rotation: 5,
                            duration: gameOptions.throwSpeed * 4,
                            callbackScope: this,
                            onComplete: function (tween) {
                                this.scene.start("main_scene")
                            }
                        });
                    }
                }

            });
        }

    }

    // 游戏每一帧执行
    update(time, delta) {


        //v1.2 修改 使目标转动起来
        //this.target.angle += gameOptions.rotationSpeed;
        this.target.angle += this.currentRotationSpeed;

        // 获取旋转的刀成员
        var children = this.knifeGroup.getChildren();

        // 对于刀的每个成员
        for (var i = 0; i < children.length; i++) {

            //v1.2 修改 刀旋转的速度设置与当前速度一致
            //children[i].angle += gameOptions.rotationSpeed;
            children[i].angle += this.currentRotationSpeed;

            // 将角度转化为弧度
            var radians = Phaser.Math.DegToRad(children[i].angle + 90);

            // 再用弧度转化为相应刀的坐标
            children[i].x = this.target.x + (this.target.width / 2) * Math.cos(radians);
            children[i].y = this.target.y + (this.target.width / 2) * Math.sin(radians);
        }
        //v1.2 调整旋转角度用线性插值表示
        this.currentRotationSpeed = Phaser.Math.Linear(this.currentRotationSpeed, this.newRotationSpeed, delta / 1000);

    }

}
```

## 小结

在原有游戏的基础上：
- 调整了速度
- 加入计分板
- 重抄 部分代码

学到了：
- 中心模式的布局
- Phaser.Math 有好多东西
- window.onload()
- 游戏场景类，的基本模式