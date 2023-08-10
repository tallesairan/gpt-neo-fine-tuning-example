# GPT-Neo-2.7B & GPT-J-6B Fine-Tuning Examples Using HuggingFace & DeepSpeed

[![medium](https://aleen42.github.io/badges/src/medium.svg)](https://medium.com/geekculture/fine-tune-eleutherai-gpt-neo-to-generate-netflix-movie-descriptions-in-only-47-lines-of-code-40c9b4c32475)
![Python3.8.6](https://img.shields.io/badge/Python-3.8.6-blue.svg)
![PyTorch1.8.1](https://img.shields.io/badge/PyTorch-1.8.1-yellow.svg)

### Installation
```sh
export HF_DATASETS_CACHE="/opt/text-generation-dev/config/models"

conda create --name gptneo python=3.9
pip install -r requirements.txt 
pip install deepspeed==0.6.5
```

## Build & Run Docker Image
```bash
docker build -t tallesairan/gpt_neo_ds:latest .
docker run --gpus all -it gpt_neo_ds bash
```

## GPT-Neo

[**Example with GPT-Neo-1.3B without DeepSpeed**](https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/gpt_neo.py)  
[**Training and testing log with GPT-Neo-1.3B**](https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/training_and_results_gpt_neo_13.txt)  
[**Example with GPT-Neo-2.7B with DeepSpeed**](https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/gpt_neo_xl_deepspeed.py)  
[**DeepSpeed configuration with GPT-Neo-2.7B**](https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/ds_config_gpt_neo_27.json)  
[**Training and testing log with GPT-Neo-2.7B**](https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/training_and_results_gpt_neo_27.txt)  

### GPU VRAM load during GPT-Neo-2.7B training
<img src="https://raw.githubusercontent.com/dredwardhyde/gpt-neo-fine-tuning-example/main/vram_gpt_neo_27.png" width="737"/>  

### RAM load during GPT-Neo-2.7B training
<img src="https://raw.githubusercontent.com/dredwardhyde/gpt-neo-fine-tuning-example/main/ram_gpt_neo_27.png" width="737"/>  

### Results  
<img src="https://raw.githubusercontent.com/dredwardhyde/gpt-neo-fine-tuning-example/main/results.png" width="1000"/>  

## GPT-J-6B

[**Example with GPT-J-6B with DeepSpeed**](https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/gpt_j_deepspeed.py)  
[**DeepSpeed configuration with GPT-J-6B**](https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/ds_config_gpt_j.json)  
[**Training and testing log with GPT-J-6B**](https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/training_and_results_gpt_j.txt)  
### GPU VRAM load during GPT-J-6B training
<img src="https://raw.githubusercontent.com/dredwardhyde/gpt-neo-fine-tuning-example/main/vram_gpt_j.png" width="737"/>  

### RAM load during GPT-J-6B training
<img src="https://raw.githubusercontent.com/dredwardhyde/gpt-neo-fine-tuning-example/main/ram_gpt_j.png" width="737"/>  