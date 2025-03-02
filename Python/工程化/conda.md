
## _基本用法_

```bash
# (1) 检查&更新
conda -V
conda update conda


# (2) 建立&激活&删除
conda env list
# --name
conda create -n py39 python=3.9
conda create -n py39 python=3.9 ipykernel numpy

# 多种方式
activate myenv
# 有时激活 base 环境，才能 conda activate
source activate
conda activate myenv
source activate myenv
# 
deactivate
source deactivate
conda env remove --name myenv
conda remove -n myenv --all


# (3) 包管理
conda list
conda list -n env_name

conda install numpy
conda install -n env_name numpy

conda remove --name myenv numpy

# 删除整个环境
conda remove --name myenv --all -y


# (4) requirements.txt环境导出&创建
pip freeze > requirements.txt
pip install -r requirements.txt

conda env export > environment.yml
```


</br>

## _channel 相关_

```bash
conda config --show channels

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge

conda config --remove channels some-channel
```



</br>

## _! 与 %_

打通kernel, conda, !, %run {}

`!cmd` : 新建一个子shell 执行 cmd, cmd 执行完，这个子 shell 也就消失了

要想在当前的shell 生效，需要使用 `%cmd`

```python
# 可以看到使用 ！的环境会有问题，而下面这行不会（中间有个自动的转魔法方法）
!pip show torch
pip show torch
```

https://jakevdp.github.io/PythonDataScienceHandbook/01.05-ipython-and-shell-commands.html

demo：

```python
prepare_command = f"./model/stylegan/prepare_data.py --out ./data/{project_name}/lmdb/ --n_worker 4 --size 1024 ./data/{project_name}/images/"

%run {prepare_command}
```

```python
# line mode
%timeit func()

# cell mode
%%timeit

func()
```

https://jakevdp.github.io/PythonDataScienceHandbook/01.05-ipython-and-shell-commands.html



</br>

## _kernel_


（1）查看kernel

```bash
pip install jupyter
python -m jupyter kernelspec list
```


（2）将虚拟环境写入 jupyter 的 kernel 上：

```bash
pip install ipykernel
python -m ipykernel install --user --name=cu118_kernel
```

```bash
pip install jupyter

jupyter notebook
```

（3）删除

```bash
python -m jupyter kernelspec remove cu118_kernel -y
```








</br>

## _other_


（1）conda-pack 环境打包：

```bash
# 多种方式
conda install conda-pack
conda install -c conda-forge conda-pack
pip install conda-pack
pip install git+https://github.com/conda/conda-pack.git


conda pack -n 环境名称 -o 环境名称.tar.gz
```

```bash
source activate example

# Package the current environment
conda-pack

ls
# example.tar.gz

# Get the file size
du -h example.tar.gz

# Unpack the environment
mkdir myenv
tar -xf example.tar.gz -C myenv

# Activate the environment
source myenv/bin/activate

# Use applications in the environment
which ipython

# Deactivate the environment
source myenv/bin/deactivate
```

参考：[conda-pack 官网](https://conda.github.io/conda-pack/)


（2）conda-forge

conda-forge 是一个社区驱动的 conda 包管理器的镜像源
