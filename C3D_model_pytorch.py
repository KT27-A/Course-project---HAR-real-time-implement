import torch
import torch.nn as nn

class C3D(nn.Module):

    def __init__(self, dropout_rate):
        super(C3D, self).__init__()
        
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 101)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.relu = nn.ReLU()

    def forward(self, x):
        # input x: 10*16*3*112*112              N C D H W           suppose batch_size = 10
        h = self.relu(self.conv1(x))                        # h: 10*64*16*112*112
        h = self.pool1(h)                                   # h: 10*64*16*56*56

        h = self.relu(self.conv2(h))                        # h: 10*128*16*56*56
        h = self.pool2(h)                                   # h: 10*128*8*28*28

        h = self.relu(self.conv3a(h))                       # h: 10*256*8*28*28
        h = self.relu(self.conv3b(h))                       # h: 10*256*8*28*28
        h = self.pool3(h)                                   # h: 10*256*4*14*14

        h = self.relu(self.conv4a(h))                       # h: 10*512*4*14*14
        h = self.relu(self.conv4b(h))                       # h: 10*512*4*14*14
        h = self.pool4(h)                                   # h: 10*512*2*7*7

        h = self.relu(self.conv5a(h))                       # h: 10*512*2*7*7
        h = self.relu(self.conv5b(h))                       # h: 10*512*2*7*7
        h = self.pool5(h)                                   # h: 10*512*1*4*4

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)
        h = self.fc8(h)                                     # h: 10*101

        
        return h
