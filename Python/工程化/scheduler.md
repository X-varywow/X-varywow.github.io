

```bash
pip install apscheduler
```


- 触发器：
  - data
  - interval
  - cron 周期触发
- 任务存储器
- 执行器
  - ThreadPoolExecutor
  - ProcessPoolExecutor
- 调度器


## _BlockingScheduler_

阻塞式调度器，程序中只运行定时任务

```python
from apscheduler.schedulers.blocking import BlockingScheduler    # 引入模块


def task():
    '''定时任务'''
    os.system('python3 spider.py')


if __name__ == '__main__':
    scheduler = BlockingScheduler()

    # 添加任务
    scheduler.add_job(task, 'cron', hour=11, minute=30)

    scheduler.start()
```

## _BackgroundScheduler_

调度器在后台独立运行

demo: 配合 fastapi 使用，定期读取配置表

```python
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI

app = FastAPI()

@app.on_event('startup')
def start_job():
    scheduler.start()
    scheduler.add_job(
        func=refresh_hot_config,
        trigger='interval',
        minutes=10
    )
```













还有更多的调度器：AsyncIOScheduler，QtScheduler。。。


-----------

参考资料：
- [python定时任务最强框架APScheduler详细教程](https://zhuanlan.zhihu.com/p/144506204)
- https://juejin.cn/post/6844903823941730311