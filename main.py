import wandb
import torch
import random
import argparse
import time  
import utils
import config
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from metrics import accuracy 
import models
from model import ViT
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from scheduler import WarmupCosineSchedule
from torchvision import datasets, transforms
from dataloader import ImageFolder
from torch.utils.data import DataLoader
import os 


parser = argparse.ArgumentParser(description='vision transformer')
parser.add_argument('--data', type=str, default='imagenet', metavar='N',
                    help='data')
parser.add_argument('--data_path', type=str, default='./path', metavar='N',
                    help='data') 
parser.add_argument('--model', type=str, default='vit', metavar='N',
                    help='model')
parser.add_argument('--batch_size', type=int, default=4096, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0.3, metavar='M',
                    help='adam weight_decay (default: 0.5)')
parser.add_argument('--t_max', type=float, default=80000, metavar='M',
                    help='cosine annealing steps')                    
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--decay_type', type=str, default='None',
                    help='warmup scheduler')
parser.add_argument('--alpha', type=int, default=0.9, metavar='N',
                    help='alpha')
parser.add_argument('--warmup_steps', type=int, default=8, metavar='N',
                    help='warmup_steps')
parser.add_argument('--world_size', type=int, default=4, metavar='N',
                    help='world_size')
parser.add_argument('--workers', type=int, default=4, metavar='N',
                    help='num_workers')                                         
parser.add_argument('--gpu',type=str,default='0',
                    help = 'gpu')
parser.add_argument('--mode',type=str,default='None',
                    help = 'train/val mode')                    
               

args = parser.parse_args()

os.environ["WANDB_API_KEY"] = ' '


cudnn.benchmark=True 

def main(rank,args) : 
    init_process(rank,args.world_size)
    
    if dist.get_rank() ==0:
        wandb.init(project = 'vision transformer',
                  )
        wandb.config.update(args)                             
    else : 
        wandb.init(project = 'vision transformer',
                   mode = 'disabled')
        wandb.config.update(args)
    
    #data load. 
    st = time.time()
    train_dir = os.path.join(args.data_path,'train')
    test_dir = os.path.join(args.data_path,'val')

    trainset = ImageFolder(
        train_dir,
        transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])]))
    valset = ImageFolder(
        test_dir,
        transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])) 
    if dist.get_rank() == 0 : 
        print('data load',time.time()-st)
    
    #DDP 
    args.batch_size = int(args.batch_size / args.world_size)
    args.workers = int((args.workers + args.world_size - 1) / args.world_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,
                                                                    rank=rank,
                                                                    num_replicas=args.world_size,
                                                                    shuffle=True)
    
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,sampler=train_sampler)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    model = ViT(in_channels = 3,
            patch_size = 16,
            emb_size = 768,
            img_size = 224,
            depth = 12,
            n_classes = 1000,
            )
    torch.cuda.set_device(rank)

    if args.mode == 'train':
        model = model.cuda(rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        wandb.watch(model)

        optimizer = optim.AdamW( model.parameters(), lr=args.lr,weight_decay = args.weight_decay)

        #learning rate warmup of 80k steps
        t_total = ( len(train_loader.dataset) / args.batch_size ) * args.epochs  
        print(f"total_step : {t_total}")

        scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=0)
        criterion = nn.CrossEntropyLoss().cuda(rank)
        
        #clip_norm
        max_norm = 1 

        #train 
        scaler = torch.cuda.amp.GradScaler()

        model.train()
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)
            for batch_idx, (data, target) in enumerate(train_loader):
                st = time.time()
                data, target = data.cuda(dist.get_rank()), target.cuda(dist.get_rank())
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output,target) 
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                scaler.step(optimizer)
                scaler.update() 
                scheduler.step()  #iter
                train_loss = reduce_tensor(loss.data,dist.get_world_size())

                if dist.get_rank() == 0 : 
                    wandb.log({'train_batch_loss' : train_loss.item()})
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), int(len(train_loader.dataset)/args.world_size),
                        100. * batch_idx / len(train_loader), train_loss))
                    print('teacher_network iter time is {0}s'.format(time.time()-st))
            if dist.get_rank() == 0 :
                print('save checkpoint') 
            
                if epoch % 15 == 0:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()
                    },filename=f'checkpoint_{epoch}.pth.tar') 
            
            model.eval()
            correct = 0
            total_acc1 = 0
            total_acc5 = 0
            step=0
            st = time.time()
            for batch_idx,(data, target) in enumerate(val_loader) :
                with torch.no_grad() :
                    data, target = data.cuda(dist.get_rank()), target.cuda(dist.get_rank())
                    output = model(data)
                val_loss = criterion(output,target) 
                val_loss = reduce_tensor(val_loss.data,dist.get_world_size())
                
                if dist.get_rank() == 0 : 
                        wandb.log({'val_batch_loss' : val_loss.item()})

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                total_acc1 += acc1[0] 
                total_acc5 += acc5[0]
                step+=1
            if dist.get_rank() == 0 :    
                print(f"[{batch_idx * len(data)}/{int(len(val_loader.dataset)/args.world_size)}, top1-acc : {acc1[0]}, top5-acc : {acc5[0]}]")
            
            if dist.get_rank() == 0 :
                print('\nval set: top1: {}, top5 : {} '.format(total_acc1/step, total_acc5/step))
                wandb.log({'top1' : total_acc1/step})
                wandb.log({'top5' : total_acc5/step})
            print(f"validation time : {time.time()-st}")

    if args.mode == 'val' : 
        checkpoint = torch.load('ckpt path')
        model = model.cuda(rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        model.load_state_dict(checkpoint['state_dict'])

        optimizer = optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay) 
        optimizer.load_state_dict(checkpoint['optimizer'])

        criterion = nn.CrossEntropyLoss().cuda(rank) 
    
        model.eval()
        correct = 0
        total_acc1 = 0
        total_acc5 = 0

        st = time.time()
        for batch_idx,(data, target) in enumerate(val_loader) :
            with torch.no_grad() :
                data, target = data.cuda(dist.get_rank()), target.cuda(dist.get_rank())
                output = model(data)
            val_loss = criterion(output,target) 
            val_loss = reduce_tensor(val_loss.data,dist.get_world_size())
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            total_acc1 += acc1[0] 
            total_acc5 += acc5[0]

        if dist.get_rank() == 0 :     
            print(f"[{batch_idx * len(data)}/{int(len(val_loader.dataset)/args.world_size)}, top1-acc : {acc1[0]}, top5-acc : {acc5[0]}]")
            
        if dist.get_rank() == 0 :
            print('\nval set: top1: {}, top5 : {} '.format(
                    experiment.log_metric(torch.mean(total_acc1)), experiment.log_metric(torch.mean(total_acc5))))
        
        print(f"validation time : {time.time()-st}")
    
    cleanup()
    wandb.finish()



def init_process(rank, world_size,backend='nccl'):
    os.environ['MASTER_ADDR'] = ''
    os.environ['MASTER_PORT'] = ''
     
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"DDP process initialized [{rank + 1}/{world_size}] rank : {rank}.")

def cleanup():
    dist.destroy_process_group()

def reduce_tensor(tensor, world_size): 
    rt = tensor.clone()
    dist.all_reduce(rt, op = dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)



if __name__ == '__main__' :  
   mp.spawn(main, nprocs = args.world_size, args = (args,))
