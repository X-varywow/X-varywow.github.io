

https://github.com/yeemachine/kalidokit/tree/main

> 这个项目已经弃用，方案集成到了 google mediapipe，作者 google 的


文件结构：
- src
  - facesolver
  - handsolver (每个配置项标定 xyz)
  - posesolver
  - utils
  - index.html 入口




## 实现细节

- index.html
  - live2dsdk
  - pixijs renderer 【用来 2D 渲染的】
  - mediapipe
    - face_mesh
    - drawing_utils
    - camera_utils
  - kalidokit 利用点位计算了特征
- script.js
- style.css
- hiyori
  - 存放 live2d 模型
  

### （1）创建一个相机，用于实时捕获，并发给 facemesh

```js
const startCamera = () => {
    const camera = new Camera(videoElement, {
        onFrame: async () => {
            await facemesh.send({ image: videoElement });
        },
        width: 640,
        height: 480
    });
    camera.start();
};
```

### （2）facemesh 定义

```js
facemesh = new FaceMesh({
    locateFile: file => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
    }
});

facemesh.onResults(onResults);
```

### （3）根据返回的 facemesh 进行更新

```js
const onResults = results => {
    drawResults(results.multiFaceLandmarks[0]);
    animateLive2DModel(results.multiFaceLandmarks[0]);
};
```

### （4）drawResults

```js
const drawResults = points => {
    if (!guideCanvas || !videoElement || !points) return;
    guideCanvas.width = videoElement.videoWidth;
    guideCanvas.height = videoElement.videoHeight;
    let canvasCtx = guideCanvas.getContext("2d");
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, guideCanvas.width, guideCanvas.height);
    // Use `Mediapipe` drawing functions
    drawConnectors(canvasCtx, points, FACEMESH_TESSELATION, {
        color: "#C0C0C070",
        lineWidth: 1
    });

    // 可见返回的点 478个，mediapipe 还是牛啊，以前用的dlib68...
    if (points && points.length === 478) {
        // 瞳孔：468， 472
        drawLandmarks(canvasCtx, [points[468], points[468 + 5]], {
            color: "#ffe603",
            lineWidth: 2
        });
    }
};
```

### （5）animateLive2DModel

```js
const animateLive2DModel = points => {
    if (!currentModel || !points) return;

    let riggedFace;

    if (points) {
        // use kalidokit face solver
        riggedFace = Face.solve(points, {
            runtime: "mediapipe",
            video: videoElement
        });
        rigFace(riggedFace, 0.5);
    }
};
```

rigFace，利用返回的特征信息驱动 live2d，如：

```js
coreModel.setParameterValueById(
    "ParamEyeBallX",
    lerp(
        result.pupil.x,
        coreModel.getParameterValueById("ParamEyeBallX"),
        lerpAmount
    )
);
```




### （6）深入 Kalidokit， 如何利用关键点，计算出配置项的信息

但是 python 没有显示 live2d 的接口，所以还得用前端显示. 基础自由度 + 放缩自由度 + 其他自由度，，，

传入
```js
const {
    Face,
    Vector: { lerp },
    Utils: { clamp }
} = Kalidokit;

// 这应该是 输入
riggedFace = Face.solve(points, {
    runtime: "mediapipe",
    video: videoElement
});
```

参考：https://github.com/yeemachine/kalidokit

但是 html 中这样引入就可以了。（两份代码大致一致，仓库代码容易理解些）

```html
<script src="https://cdn.jsdelivr.net/npm/kalidokit@1.1/dist/kalidokit.umd.js"></script>
```

`lerp` linear interpolation, 两个数值之间按照一定比例进行插值计算，得到一个中间值。

`文件结构`(都在 src 下)
- facesolver
- handsolver (每个配置项标定 xyz)
- posesolver
- utils
  - euler.ts (欧拉旋转)
  - healper.ts (保存默认姿态配置)
  - vector.ts (几何运算，东西很多)
- constants.ts
- index.ts (全部功能 export)
- type.ts （定义数据类型）


### （7）demo: getEyeOpen

```js
/**
 * Calculate eye open ratios and remap to 0-1
 * @param {Array} lm : array of results from tfjs or mediapipe
 * @param {Side} side : designate left or right
 * @param {Number} high : ratio at which eye is considered open
 * @param {Number} low : ratio at which eye is comsidered closed
 */
export const getEyeOpen = (lm: Results, side: Side = LEFT, { high = 0.85, low = 0.55 } = {}) => {
    const eyePoints = points.eye[side];  // [130, 133, 160, 159, 158, 144, 145, 153]

    const eyeDistance = eyeLidRatio(
        lm[eyePoints[0]],
        lm[eyePoints[1]],
        lm[eyePoints[2]],
        lm[eyePoints[3]],
        lm[eyePoints[4]],
        lm[eyePoints[5]],
        lm[eyePoints[6]],
        lm[eyePoints[7]]
    );

    // human eye width to height ratio is roughly .3
    const maxRatio = 0.285;
    // compare ratio against max ratio
    const ratio = clamp(eyeDistance / maxRatio, 0, 2);
    // remap eye open and close ratios to increase sensitivity
    const eyeOpenRatio = remap(ratio, low, high);
    return {
        // remapped ratio
        norm: eyeOpenRatio,
        // ummapped ratio
        raw: ratio,
    };
};
```

```js
// eyeLidRatio 

//use 2D Distances instead of 3D for less jitter
const eyeWidth = (eyeOuterCorner as Vector).distance(eyeInnerCorner as Vector, 2);
const eyeOuterLidDistance = (eyeOuterUpperLid as Vector).distance(eyeOuterLowerLid as Vector, 2);
const eyeMidLidDistance = (eyeMidUpperLid as Vector).distance(eyeMidLowerLid as Vector, 2);
const eyeInnerLidDistance = (eyeInnerUpperLid as Vector).distance(eyeInnerLowerLid as Vector, 2);
const eyeLidAvg = (eyeOuterLidDistance + eyeMidLidDistance + eyeInnerLidDistance) / 3;
const ratio = eyeLidAvg / eyeWidth;

return ratio;

```







> 其实这个少了很多自由度：眉毛，转向幅度不够大，没低头动作等等。



## 框架搭建，flask + live2d web



## 一些问题？

只根据点驱动就行了吗？看看mediapipe 这个功能，其他功能


## 0807 try:

似乎明白了，通过 `coreModel.setParameterValueById` 和 运动计算，来添加视频动捕的自由度

通过 lerp 线性插值 来实现动画的连贯


```js
const {
    Application,
    live2d: { Live2DModel }
} = PIXI;
```

解构语法，就是引用？ PIXI 下 live2d 的 Live2DModel 、PIXI 下Application 可以直接使用了

```js
// 这是 mediapipe 检测出的字典
result.head
```

**Face,solve 即 kalidokit 会根据检测出的点位计算出特征，如 head.degrees.z**

下一步：
- 知道 res 的格式
- live2d 基础
- 两端打通