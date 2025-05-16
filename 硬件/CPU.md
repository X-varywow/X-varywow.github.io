

## other

`0.5c 实现`

机器资源 0.5c, 是通过 **CPU配额限制** 或 **时间片共享** 实现的。

1s 内， 0.5c 的实例只能占用 CPU 500ms.

```bash
# 查看 CPU 配额（适用于容器）
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us  # 可能显示 50000（即 50ms/100ms，表示 0.5c）
cat /sys/fs/cgroup/cpu/cpu.cfs_period_us # 通常 100000（100ms）

# 测试 CPU 算力（对比 1c 和 0.5c 的差异）
stress -c 1  # 单核压力测试
top          # 观察 CPU 使用率是否被限制在 50%
```

-------------

`超线程`

超线程（Hyper-Threading，简称 HT）是 Intel 开发的一项 CPU 技术（AMD 的类似技术叫 SMT，Simultaneous Multi-Threading），它允许 单个物理 CPU 核心同时执行多个线程，从而提高 CPU 的并行计算能力。