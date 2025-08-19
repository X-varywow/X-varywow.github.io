vscode:
- word count
- indent-rainbow
- 设置：自动保存
- workbench>tree indent: 20
- audio: 0
- Editor: Format On Save


> 使用 console.log(document.cookie) 可直接获取网页的 cookie

插件 EditorConfig for VS Code 统一代码样式

字体：[FiraCode](https://github.com/tonsky/FiraCode)


## _插件_

ruff / pylance 插件发现基础的语法错误;

Jupyter 扩展可替换 Jupyter Notebook;

-----------------

gitlens, source control panel and detach commits view;

commits 历史修改对比

分支对比：（gitlens inspect） -> (compare references)

文件行级别最近修改：右上角 file annotations

default date format:  "YYYY/MM/DD HH:mm"

diff 比较时没显示行首的空格差异：右上角有个图标可开启这个功能


## _.vscode_

.vscode 下会有各项的 json 配置
- extensions.json
- launch.json
- settings.json
- tasks.json


### settings.json

```json
{
    // "window.zoomLevel": 0.3,
    // "editor.fontSize": 15,
    "editor.tabSize": 4,
    "editor.formatOnSave": true,
    "notebook.formatOnSave.enabled": true,
    // "python.languageServer": "None", // 禁用 Pylance, 会导致无法跳转
    // cursor 自带的 python 扩展，可支持跳转，同时 basedpyright 检查
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit",
            "source.fixAll": "explicit"
        }
    },
    "black-formatter.args": [
        "--line-length",
        "120",
        "--experimental-string-processing",
        "--skip-magic-trailing-comma"
    ],
    "isort.args": [
        "--profile",
        "black"
    ],
    // ruff 替代 pylint
    "ruff.lint.enable": true,
    "ruff.nativeServer": "on",
    // "ruff.format.args": [
    //     "--line-length",
    //     "120",
    //     "--experimental-string-processing",
    //     "--skip-magic-trailing-comma"
    // ],
    // pylance pylint 都提供基本代码检查，pylance 提供风格检查
    "pylint.enabled": false,
    "pylint.args": [
        "--disable=C0116",
        "--disable=C0115",
        "--disable=C0114",
        "--disable=C0303",
        "--disable=C0112",
        "--disable=W0611",
        "--disable=W0621",
        "--disable=W0702",
        "--disable=W0718",
    ]
}
```

看明细扩展的介绍可以了解到对应设置


### launch.json 

run & debug 相关，name 会展示在左侧条目上

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



### tasks.json

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


## trae

解决与 vscode 颜色不一致: 关闭 Semantic Highlighting


line-height: 0


右键属性 -> 管理员身份运行， 防止特殊操作（python 移动鼠标不生效）

## cursor

workbench.activityBar.orientation


## 快捷键

| 按键         | 说明           |
| ------------ | -------------- |
| ctrl + `     | 打开terminal   |
| ctrl + enter | 快速开启下一行 |





| 快捷键           | 说明                     |
| ---------------- | ------------------------ |
| cmd + D          | 选定一个单词             |
| cmd + C, cmd + V |                          |
| cmd + X          | 删除行                   |
| cmd + Z          |                          |
| cmd + click      | 查看函数定义，被引用位置 |


## 调试

`Step into`

执行下一行代码，进入函数调用。


`Step over`

执行下一行代码，但不进入任何函数调用。

如果当前行是一个函数调用，调试器会执行整个函数，但不进入该函数内部。


`Step out`

用于从当前函数退出，执行完当前函数的剩余部分，并停在函数的返回点上。





--------------

参考资料：
- [PyCharm 中文指南(Win版)](https://pycharm.iswbm.com/)