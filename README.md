## Vision Transformer(Pytorch) 

Pytorch implementation of Google AI's 2021 Vision Transformer. (Distributed Data Parallel & Mixed Precision)

- Title : An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

- Link : https://arxiv.org/abs/2010.11929

##  Quickstart 

### Weights & Biases(Visualization tool)

- Before starting, you should login wandb using your personal API key. 
- Weights & Biases : https://wandb.ai/site

```shell
wandb login PERSONAL_API_KEY
```

### Cloning a repository

```shell
git clone https://github.com/ssuncheol/Pytorch-VIT.git
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
|:---------:|:--------:|:-------:|:--------:|:----------:|:---------:|:---------:|:------:|
| ViT=B=16 	| 86M | 224 	| 16 	| 768| 3072 | 12 | 12 | 



## Experiments 

I use ImageNet-1k Dataset to train and evalute model 

### Arguments
| Args 	| Type 	| Description 	| Default|
|:---------:|:--------:|:----------------------------------------------------:|:-----:|
| Epochs 	| [int] 	| Epochs | 300|
| Batch_size 	| [int] 	| Batch size| 1024|
| Model 	| [str]	| Vision Transformer| 	vit|
| Optimizer 	| [str]	| Adam, Adamw| 	adamw|
| Warnup_steps 	| [int]	| Warmup steps| 100k|
| Weight_decay 	| [float]	| Weight decay | 0.3|
| Dropout 	| [float]	| Dropout | 0.0|


### How to train

```shell
python3 main.py --lr=0.001 --batch_size=1024 --weight_decay=0.3 --mode='train' --world_size=8
```

### Result 
| --- 	| Dataset 	| Batch size | Top-1 	|
|:---------:|:--------:|:---------------------------------------:|:-----:|
| Original paper 	| ImageNet-1k 	| 4096 | 74.6% 	|
| Implementation 	| ImageNet-1k 	| 1024 | 72.58% 	| 

-  There is a difference in performance because of batch size

### Reference 
- https://arxiv.org/abs/1706.03762
- https://arxiv.org/abs/2010.11929
- https://arxiv.org/abs/2106.01548

