
## _CONDA_

### 基本用法

```bash
# (1) 检查&更新
conda -V
conda update conda


# (2) 建立&激活&删除
conda env list
# --name
conda create -n myenv python=3.5
conda create -n myenv python=3.5 ipykernel numpy

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


### channel 相关

```bash
conda config --show channels

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge

conda config --remove channels some-channel
```



### yaml

```bash
conda env export >  environment.yaml

conda env update -f environment.yaml

conda env create -f ./environment/vtoonify_env.yaml
```

```yaml
name: dualstylegan_env
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - _libgcc_mutex=0.1=conda_forge
  - ca-certificates=2022.2.1=h06a4308_0
  - certifi=2021.10.8=py38h06a4308_2    
  - faiss=1.7.1=py38h7b17aaf_0_cpu
  - libedit=3.1.20191231=he28a2e2_2
  - libfaiss=1.7.1=hb573701_0_cpu
  - libfaiss-avx2=1.7.1=h1234567_0_cpu
  - libffi=3.2.1=he1b5a44_1007
  - libgcc-ng=9.3.0=h2828fa1_19
  - libstdcxx-ng=9.3.0=h6de172a_19
  - matplotlib-base=3.3.4=py38h62a2d02_0
  - pillow=8.3.1=py38h2c7a002_0
  - pip=21.1.3=pyhd8ed1ab_0
  - python=3.8.3=cpython_he5300dc_0
  - python-lmdb=1.2.1=py38h2531618_1
  - pytorch=1.7.1=py3.8_cuda10.1.243_cudnn7.6.3_0
  - setuptools=49.6.0=py38h578d9bd_3
  - scikit-image=0.18.1=py38ha9443f7_0
  - torchaudio=0.7.2=py38
  - torchvision=0.8.2=py38_cu101
  - pip:
    - cmake==3.21.0
    - dlib==19.21.0
    - matplotlib==3.4.2
    - ninja==1.10.2    
    - numpy==1.21.0
    - opencv-python==4.5.3.56
    - scipy==1.7.0 
    - tqdm==4.61.2
    - wget==3.2
prefix: ~/anaconda3/envs/dualstylegan_env
```

</br>

### ! 与 %

打通kernel, conda, !, %run {}

`!cmd` : 新建一个子shell 执行 cmd, cmd 执行完，这个子 shell 也就消失了

要想在当前的shell 生效，需要使用 `%cmd`

```python
# 可以看到使用 ！的环境会有问题，而下面这行不会（中间有个自动的转魔法方法）
!pip show torch
pip show torch
```

https://jakevdp.github.io/PythonDataScienceHandbook/01.05-ipython-and-shell-commands.html

常见用法：

```python
# cmd 中创建虚拟环境
# 将虚拟环境正确安装到一个新的 kernel

prepare_command = f"./model/stylegan/prepare_data.py --out ./data/{project_name}/lmdb/ --n_worker 4 --size 1024 ./data/{project_name}/images/"

%run {prepare_command}

# 这可以更加工程化，可说明；相当把整个 bash 流程都可以搬到 jupyter 上。
```
pip install oscrypto git+https://github.com/wbond/oscrypto.git@d5f3437ed24257895ae1edd9e503cfb352e635a8




</br>

### kernel


（1）查看kernel

```bash
jupyter kernelspec list
```


（2）将虚拟环境写入 jupyter 的 kernel 上：

方式一：
```bash
ipython kernel install --user --name vt
```

（这个好像是从默认的环境装载 kernel, 从当前虚拟环境装载要使用方式二）


方式二：
```bash
conda install ipykernel
python -m ipykernel install --user --name=test
```

```bash
pip install jupyter

jupyter notebook
```

</br>

### other


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

（4）打通kernel, conda, !, %run {}

!cmd 新建一个子shell 执行 cmd, cmd 执行完，这个子 shell 也就消失了

要想在当前的shell 生效，需要使用 %cmd

```python
# 可以看到使用 ！的环境会有问题，而下面这行不会（中间有个自动的转魔法方法）
!pip show torch
pip show torch
```

https://jakevdp.github.io/PythonDataScienceHandbook/01.05-ipython-and-shell-commands.html

常见用法：

```python
# cmd 中创建虚拟环境
# 将虚拟环境正确安装到一个新的 kernel

prepare_command = f"./model/stylegan/prepare_data.py --out ./data/{project_name}/lmdb/ --n_worker 4 --size 1024 ./data/{project_name}/images/"

%run {prepare_command}

# 这可以更加工程化，可说明；相当把整个 bash 流程都可以搬到 jupyter 上。
```

colab 中有这种语法：

```python
vcmd= f"checkout {commit}"
!git $vcmd
```


## _Poetry_

python packaging and dependency management made easy.

在 pip 基础上新增了依赖性管理

poetry 使用的 pyproject.toml 是 PEP 518 所提出的新标准，相当于 npm 的 package.json

```bash
pip install poetry

# 使用 poetry 创建项目
poetry new poetry_demo

poetry add flask

# 查看 help
poetry

# other
poetry env info

```








---------

参考资料：
- [Poetry 完全入門指南](https://blog.kyomind.tw/python-poetry/)