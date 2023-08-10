# GPT-Neo-1.3B - GPT-Neo-2.7B & GPT-J-6B Fine-Tuning Examples Using HuggingFace & DeepSpeed

![Python3.8.6](https://img.shields.io/badge/Python-3.8.6-blue.svg)
![PyTorch1.8.1](https://img.shields.io/badge/PyTorch-1.8.1-yellow.svg)

### Installation
```sh
After Container Start
git clone https://github.com/microsoft/DeepSpeed.git
DS_BUILD_CPU_ADAM=1 DS_BUILD_TRANSFORMER=1 DS_BUILD_TRANSFORMER_INFERENCE=1 DS_BUILD_QUANTIZER=1 DS_BUILD_AIO=1 DS_BUILD_CPU_ADAGRAD=1 DS_BUILD_SPARSE_ATTN=0 python setup.py build_ext -j8 bdist_wheel

DS_BUILD_OPS=1 DS_BUILD_SPARSE_ATTN=0 pip install deepspeed
```

## Build

```bash
docker build -t tallesairan/gpt_neo_ds:latest .
docker run --gpus all -it gpt_neo_ds bash
deepspeed --num_gpus=4 gpt_neo_27_ds_blog_posts.py --deepspeed ds_config_gpt_neo_27.json
```

## Run
```bash
deepspeed --num_gpus=4 gpt_neo_27_ds_blog_posts.py --deepspeed ds_config_gpt_neo_27.json
```
 
### Credits

[![medium](https://aleen42.github.io/badges/src/medium.svg)](https://medium.com/geekculture/fine-tune-eleutherai-gpt-neo-to-generate-netflix-movie-descriptions-in-only-47-lines-of-code-40c9b4c32475)
  

