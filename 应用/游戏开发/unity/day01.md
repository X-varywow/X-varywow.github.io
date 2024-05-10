初步学一下 unity, c#；完成 [一个文字放置游戏](https://zhuanlan.zhihu.com/p/100822441)

- 学会使用 UI，编辑器使用；
- Btn、text 控件
- 初级代码编写，利用脚本操纵结点
- 拖动操作实现物体、事件、属性的绑定


```cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class NewBehaviourScript : MonoBehaviour
{
    public Text Text_News, Text_News_Gain;
    public Text Text_FPS;
    int PuTao, JingYan;


    public void Button_addExp_F(){
        if(PuTao > 0){
            PuTao --;
            JingYan ++;
            Text_News.text = "成功获取，当前经验：" + JingYan;
        }else{
            Text_News.text = "等待收益";
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        Application.targetFrameRate = 60;
    }

    // Update is called once per frame
    void Update()
    {
         float FPS = 1 / Time.deltaTime;
         Text_FPS.text = "FPS: " + FPS;
         PuTao ++;

         Text_News_Gain.text = "当前拥有葡萄：" + PuTao;

    }
}
```

在 start 中 添加 Application.targetFrameRate = 60; 限定帧率为 60