""" How to use C3D network. """
import numpy as np
import argparse
import torch
import os
from os.path import join
import time
import datetime
import skimage.io as io
from skimage.transform import resize
import data_processing

import C3D_model_pytorch as D
from torch.optim import lr_scheduler
# ----------------------------------------
#        Initialize the parameters
# ----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=20, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='SGD: learning rate')
parser.add_argument('--checkpoint_interval', type=int, default=10, help='interval between model checkpoints')
parser.add_argument('--num_classes', type=int, default=101, help='num of output classes')
opt = parser.parse_args()
print(opt)
cuda = True if torch.cuda.is_available() else False

TRAIN_LOG_DIR = os.path.join('Log/train/', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
TRAIN_CHECK_POINT = 'check_point/'
TRAIN_LIST_PATH = 'train.list'
# TEST_LIST_PATH = 'test.list'
# TRAIN_LIST_PATH = 'train_cross.list'

# ----------------------------------------
#       Network training parameters
# ----------------------------------------
# Loss functions
criterion = torch.nn.CrossEntropyLoss()

# Initialize model
model = D.C3D(dropout_rate=0.5)

if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

# Optimizers
# optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr,weight_decay=0.0005)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0.0005)
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.1)
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# ----------------------------------------
#           Data preprocessing
# ----------------------------------------

#Get shuffle index
train_video_indices = data_processing.get_video_indices(TRAIN_LIST_PATH)

# ----------------------------------------
#                 Training
# ----------------------------------------

torch.backends.cudnn.benchmark=True

for epoch in range(opt.epochs):
    # scheduler.step()
    accuracy_epoch = 0
    loss_epoch = 0
    batch_index = 0
    step = 0
    for i in range(len(train_video_indices) // opt.batch_size):
        step += 1
        batch_data, batch_index = data_processing.get_batches(TRAIN_LIST_PATH, opt.num_classes, batch_index,
                                                                train_video_indices, opt.batch_size)
        data = batch_data['clips']
        data = data.transpose(0, 4, 1, 2, 3)
        data = torch.from_numpy(data)
        data = data.type(torch.cuda.FloatTensor)
        target = batch_data['labels']
        target = torch.from_numpy(target)
        target = target.type(torch.cuda.LongTensor)
        if cuda:
            data = data.cuda()
            target = target.cuda()
        
        optimizer.zero_grad()

        out = model(data)
        loss = criterion(out, target)

        loss.backward()
        optimizer.step()
        accuracy = np.mean(np.equal(torch.argmax(out, 1).cpu().numpy(), batch_data['labels']))
        print('Epoch %d, Batch %d, Loss is %.5f, accuracy is %f'%(epoch+1, opt.batch_size*(i+1), loss.item(), accuracy))
        
    if epoch % opt.checkpoint_interval == 0 and epoch > 1:
        # Save model checkpoints
        torch.save(model.state_dict(), 'C3D_model_pytorch.pkl')
