
[Vue live real-time avatar from your webcam in the browser](https://vuejsexamples.com/vue-live-real-time-avatar-from-your-webcam-in-the-browser/)

[Vitar 仓库](https://github.com/LarchLiu/vitar)

[实时动捕体验](https://mediapipe-studio.webapps.google.com/demo/face_landmarker)


网页端，实现对动作的更新，以形变器的形式：

```js
// Adding 0.3 to ParamMouthForm to make default more of a "smile"
coreModel.setParameterValueById(
    "ParamMouthForm",
    0.3 +
    lerp(
        result.mouth.x,
        coreModel.getParameterValueById("ParamMouthForm"),
        0.3
    )
);
console.log(JSON.stringify(result))
```


## 使用教程

```js
<head>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.js"
    crossorigin="anonymous"></script>
</head>
```