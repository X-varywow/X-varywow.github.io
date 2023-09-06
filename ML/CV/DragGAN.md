

https://github.com/XingangPan/DragGAN


```bash
# before you build the docker container, make sure you have cloned this repo, and downloaded the pretrained model by `python scripts/download_model.py`.
docker build . -t draggan:latest  
docker run -p 7860:7860 -v "$PWD":/workspace/src -it draggan:latest bash
# (Use GPU)if you want to utilize your Nvidia gpu to accelerate in docker, please add command tag `--gpus all`, like:
#   docker run --gpus all  -p 7860:7860 -v "$PWD":/workspace/src -it draggan:latest bash

cd src && python visualizer_drag_gradio.py --listen
```