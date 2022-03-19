# Vision Transformer-pytorch 

Pytorch implementation of Google AI's 2021 Vision Transformer. 

- Title : An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

- Link : https://arxiv.org/abs/2010.11929



##  Quickstart 

### Wandb 

Before starting, you should login wandb using your personal API key. 

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

### 
