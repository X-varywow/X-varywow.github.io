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


## yaml

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