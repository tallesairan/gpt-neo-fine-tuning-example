# GPT-Neo-2.7B & GPT-J-6B Fine-Tuning Examples Using HuggingFace & DeepSpeed

[![medium](https://aleen42.github.io/badges/src/medium.svg)](https://medium.com/geekculture/fine-tune-eleutherai-gpt-neo-to-generate-netflix-movie-descriptions-in-only-47-lines-of-code-40c9b4c32475)
![Python3.8.6](https://img.shields.io/badge/Python-3.8.6-blue.svg)
![PyTorch1.8.1](https://img.shields.io/badge/PyTorch-1.8.1-yellow.svg)

### Installation
```sh
cd venv/bin
./pip install -r ../../requirements.txt 
./pip install deepspeed==0.5.9
```

## GPT-Neo

[**Example with GPT-Neo-2.7B with DeepSpeed**](https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/gpt_neo_xl_deepspeed.py)  
[**Example with GPT-Neo-1.3B without DeepSpeed**](https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/gpt_neo.py)  
[**DeepSpeed configuration with GPT-Neo-2.7B**](https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/ds_config.json)  
[**Training and testing log with GPT-Neo-2.7B**](https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/training_and_results.txt)  
### GPU VRAM load during GPT-Neo-2.7B training
<img src="https://raw.githubusercontent.com/dredwardhyde/gpt-neo-fine-tuning-example/main/gpu.png" width="737"/>  

### Results  
<img src="https://raw.githubusercontent.com/dredwardhyde/gpt-neo-fine-tuning-example/main/results.png" width="1000"/>  

## GPT-J-6B

[**Example with GPT-J-6B with DeepSpeed**](https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/gpt_j_deepspeed.py)  
[**DeepSpeed configuration with GPT-Neo-2.7B**](https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/ds_config_gpt_j.json)  
### GPU VRAM load during GPT-J-6B training
<img src="https://raw.githubusercontent.com/dredwardhyde/gpt-neo-fine-tuning-example/main/vram_gpt_j.png" width="737"/>  

### RAM load during GPT-J-6B training
<img src="https://raw.githubusercontent.com/dredwardhyde/gpt-neo-fine-tuning-example/main/ram_gpt_j.png" width="737"/>  