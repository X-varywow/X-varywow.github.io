vscode:
- word count
- indent-rainbow
- 设置：自动保存
- workbench>tree indent: 20
- audio: 0
- Editor: Format On Save
- LCPR: leetcode

> 使用 console.log(document.cookie) 可直接获取网页的 cookie

插件 EditorConfig for VS Code 统一代码样式


</br>

## _插件_


gitlens, source control panel and detach commits view;


pylance 插件发现基础的语法错误;

Jupyter 扩展可替换 Jupyter Notebook;

autopep8 可以 format on save


-----------------

gitlens

commits 历史修改对比

分支对比：（gitlens inspect） -> (compare references)

文件行级别最近修改：右上角 file annotations

default date format:  "YYYY/MM/DD HH:mm"



</br>

## _.vscode_

.vscode 下会有各项的 json 配置
- extensions.json
- launch.json
- settings.json
- tasks.json


settings.json

```json
{
    "window.zoomLevel": 1,
    "editor.tabSize": 4,
    "editor.formatOnSave": true
}
```

launch.json （run & debug 相关信息）

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "APP_ENV": "test"
            }
        }
    ]
}
```

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "启动 Electron",
      "type": "node",
      "request": "launch",
      "cwd": "${workspaceFolder}",
      "runtimeExecutable": "npm",
      "windows": {
        "runtimeExecutable": "npm.cmd"
      },
      "args": ["start"],
      "outputCapture": "std"
    }
  ]
}
```



tasks.json

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "启动 Electron 应用",
      "type": "shell",
      
      // 执行命令，会从 package.json 定义的 start 启动
      "command": "npm start",

      // 定义为构建任务，设置为默认任务
      "group": { 
        "kind": "build",
        "isDefault": true
      },

      // 展示方式；总是新启面板
      "presentation": {
        "reveal": "always",
        "panel": "new"
      },
      "problemMatcher": []
    }
  ]
}
```

可使用 ctrl + shift + b 运行 task

ctrl + shift + p 打开 vscode 魔法命令行， 如 `>Tasks: Run Task`


command + shift + p `>Open User Settings (JSON))`

```json
    "terminal.integrated.fontSize": 14,
    "terminal.integrated.fontFamily": "Fira Code",
    "terminal.integrated.lineHeight": 1.2
```




字体：[FiraCode](https://github.com/tonsky/FiraCode)



## trae

解决与 vscode 颜色不一致: 关闭 Semantic Highlighting