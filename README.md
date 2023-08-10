# GPT-Neo-1.3B - GPT-Neo-2.7B & GPT-J-6B On Docker
 

### Installation
```sh
Setup DeepSpeed with your specific options
git clone https://github.com/microsoft/DeepSpeed.git
DS_BUILD_CPU_ADAM=1 DS_BUILD_TRANSFORMER=1 DS_BUILD_TRANSFORMER_INFERENCE=1 DS_BUILD_QUANTIZER=1 DS_BUILD_AIO=1 DS_BUILD_CPU_ADAGRAD=1 DS_BUILD_SPARSE_ATTN=0 python setup.py build_ext -j8 bdist_wheel

pip install deepspeed-0.10.1+0c75f4a3-cp39-cp39-linux_x86_64.whl
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
  

