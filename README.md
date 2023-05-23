## Vision Transformer : A Pytorch Implementation 

This is a Pytorch implementation of Google AI's 2021 Vision Transformer. (Distributed Data Parallel & Mixed Precision)

- Title : An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

- Link : https://arxiv.org/abs/2010.11929

## Requirements 

```shell
Cuda 11.0
Python3 3.8
PyTorch 1.8 
Torchvision 0.10.0
Einops 0.4.1
```

##  Quickstart 

### Weights & Biases(Visualization tool)

- Before starting, you should login wandb using your personal API key. 
- Weights & Biases : https://wandb.ai/site

```shell
!pip install wandb
wandb login PERSONAL_API_KEY
```

### Cloning a repository

```shell
git clone [https://github.com/ssuncheol/vision-transformer-pytorch.git]
```

### Prepare a dataset(Imagenet-1k)

- Download Imagenet-1k from open-source 

```shell 
wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar
```
- Make the folder structure of the dataset as follows 

```
train
 └─── n01440764
          └───── n01440764_18.jpeg
          └───── n01440764_36.jpeg
          └───── ...
 └─── n01443537
          └───── n01443537_2.jpeg
          └───── ...
 └─── ...

val
 └─── n01440764
          └───── ILSVRC2012_val_00000293.jpeg
          └───── ILSVRC2012_val_000002138.jpeg
          └───── ...
 └─── n01443537
          └───── ILSVRC2012_val_00000236.jpeg
          └───── ...
 └─── ...
```

### Model 

I use ViT-B-16. model's configuration as follows  

| Model 	| Parameters | Image size 	| Patch size 	| Hidden dim(patch/position embedding dim)| MLP dim | Heads(multi-head-att) | Depth(transformer blocks) | 
|:-------------:|:--------:|:-------:|:--------:|:----------:|:---------:|:---------:|:------:|
| ViT-B-16 	| 86M | 224 	| 16 	| 768| 3072 | 12 | 12 | 



## Experiments 

I use ImageNet-1k Dataset to train and evalute model 

### Arguments
| Args 	| Type 	| Description 	| Default|
|:---------:|:--------:|:----------------------------------------------------:|:-----:|
| Epochs 	| [int] 	| Epochs | 300|
| Batch_size 	| [int] 	| Batch size| 1024|
| Model 	| [str]	| Vision Transformer| 	vit|
| Optimizer 	| [str]	| Adam, Adamw| 	AdamW|
| Learning rate | [str] | Learning rate | 1e-3 |
| Weight_decay 	| [float]	| Weight decay | 0.3|
| T_max 	| [int]	| Cosine Annealing step | 80000 |
| Dropout 	| [float]	| Dropout | 0.0|
| World_size 	| [int]	| World size | 8 |


### How to train

Training time is 5 days with 8 GPUs. (RTX 3090 Ti)


```shell
python3 main.py --lr=0.001 --batch_size=1024 --weight_decay=0.3 --t_max=80000 --mode='train' --world_size=8
```

### How to eval 

```shell
python3 main.py --lr=0.001 --batch_size=1024 --weight_decay=0.3 --model='val' --world_size=8
```


### Result 
| - 	| Dataset 	| Batch size | Top-1 	|
|:---------:|:--------:|:---------------------------------------:|:-----:|
| Original paper 	| ImageNet-1k 	| 4096 | 74.6% 	|
| Implementation 	| ImageNet-1k 	| 1024 | 72.58% 	| 

-  There is a difference in performance because of batch size

### Reference 
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762 "Attention Is All You Need")
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929 "Vision Transformer")
- [When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations](https://arxiv.org/abs/2106.01548 "Vision Transformer")
- [Recent Advances in Vision Transformer: A Survey and Outlook of Recent Work](https://arxiv.org/abs/2203.01536 "Vision Transformer")

