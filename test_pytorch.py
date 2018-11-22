import C3D_model_pytorch
import data_processing
import argparse
import torch
from torch.autograd import Variable
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=10, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='SGD: learning rate')
parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between model checkpoints')
parser.add_argument('--num_classes', type=int, default=101, help='num of output classes')
opt = parser.parse_args()

def get_clip():
    pass


def read_labels_from_file(filepath):
    with open(filepath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


test_path = 'test_cross.list'
torch.backends.cudnn.benchmark=True
if __name__ == "__main__":
    with open(test_path, 'r') as t:
        test_num = len(list(t))
    test_video_indices = range(test_num)
    batch_index = 0
    for i in range(test_num // opt.batch_size):
        batch_data, batch_index = data_processing.get_batches(test_path, opt.num_classes, batch_index,
                                                                    test_video_indices, opt.batch_size)
        
        clip = torch.from_numpy(batch_data['clips'].transpose(0, 4, 1, 2, 3))
        clip = clip.cuda()
        net = C3D_model_pytorch.C3D(dropout_rate=1)
        net.load_state_dict(torch.load('C3D_model_pytorch.pkl'))
        net.cuda()
        net.eval()

        output = net(clip)

        labels = torch.from_numpy(np.array(batch_data['labels']))
        output_index = torch.max(output, 1)[1].cpu().numpy()
        equ_num = 0
        for i in range(opt.batch_size):
            if output_index[i] == labels[i]:
                equ_num += 1
        accuracy = equ_num / opt.batch_size
        print(accuracy)
        
