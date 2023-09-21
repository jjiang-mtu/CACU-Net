"""
Training Script
"""

import os
from time import time

import torch
torch.cuda.empty_cache()

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from net.CACUNet import net
from loss.multi_class_loss import DiceLoss
from dataset.dataset import train_ds


# Defining hyperparameters
on_server = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' if on_server is False else '0,1'
cudnn.benchmark = True
Epoch = 600
leaing_rate = 1e-4
alpha = 0.4 # Initial value of deep supervised attenuation factor

batch_size = 3 if on_server is False else 2
num_workers = 1 if on_server is False else 2
pin_memory = False if on_server is False else True

net = torch.nn.DataParallel(net,device_ids=[0,1]).cuda() # Specify the device to be used

# Defining data loading
train_dl = DataLoader(train_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)

# Defining the loss function
loss_func = DiceLoss()

# Defining the optimizer
opt = torch.optim.Adam(net.parameters(), lr=leaing_rate)

# Learning rate decay
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30,60,90,120,150,180,210,250,300,350,400,450,500,520,540,560,580], gamma= 0.80)

# Define the training network
start = time()
min_loss = 10

for epoch in range(Epoch):

    lr_decay.step()

    mean_loss = []
    

    for step, (ct, seg) in enumerate(train_dl):

        ct = ct.cuda()

        outputs1_stage1, outputs2_stage1, outputs3_stage1, outputs4_stage1, outputs1_stage2, outputs2_stage2, outputs3_stage2, outputs4_stage2 = net(ct)
        
        loss0 = loss_func(outputs1_stage1, outputs1_stage2, seg)
        loss1 = loss_func(outputs2_stage1, outputs2_stage2, seg)
        loss2 = loss_func(outputs3_stage1, outputs3_stage2, seg)
        loss3 = loss_func(outputs4_stage1, outputs4_stage2, seg)

        loss = loss3  +  alpha * (loss0 + loss1 + loss2) 

        mean_loss.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()       
        
        if step % 4 == 0:
            print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss.item(), (time() - start) / 60))

    mean_loss = sum(mean_loss) / len(mean_loss)
    torch.cuda.empty_cache()
    
    # Saving Models
    # The network model is named as follows: number of epoch rounds + loss of the current minibatch + average loss of this epoch round   
    
    if mean_loss < min_loss:
        min_loss = mean_loss
        print('Save best model at Epoch: {} | Loss: {}'.format(epoch, min_loss))
        torch.save(net.state_dict(), './model/Bestnet{}-{:.3f}-{:.3f}.pth'.format(epoch, loss.item(), mean_loss))

    if epoch % 50 == 0:
        torch.save(net.state_dict(), './model/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss.item(), mean_loss))

    # Attenuation of depth supervision coefficients
    if epoch % 30 == 0: alpha *= 0.8    

