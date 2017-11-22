import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
# Reference Url: https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
class FocalLoss(nn.Module):
    '''
    test:
        m = nn.Conv2d(16, 4, (3, 3)).float()
        # input is of size N x C x height x width
        input = autograd.Variable(torch.randn(3, 16, 10, 10))
        # each element in target has to have 0 <= value < C
        target = autograd.Variable(torch.LongTensor(3, 8, 8).random_(0, 4))
        loss = FocalLoss(gamma=2,alpha=1)
        output = loss(m(input), target)
        print(output)
        loss = FocalLoss(gamma=2,alpha=[1,1,1,1])
        output = loss(m(input), target)
        print(output)
    '''
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        N = input.size(0)
        C = input.size(1)

        if self.alpha is not None:
            if isinstance(self.alpha,(float,int)): 
                list_tmp = []
                for n in range(N):
                    list_alphas = []
                    for c in range(C):
                        list_alphas.append(self.alpha)
                    list_tmp.append(list_alphas)
                self.alpha = torch.Tensor(list_tmp)
                if self.alpha.type()!=input.data.type(): #convert to cuda
                    self.alpha = self.alpha.type_as(input.data)
                self.alpha = Variable(self.alpha)
            if isinstance(self.alpha,list): 
                assert len(self.alpha)==C,"Make sure len(self.alpha) compatible with C"+str(C)
                list_tmp = []
                for n in range(N):
                    list_tmp.append(self.alpha)
                self.alpha = torch.Tensor(list_tmp)
                if self.alpha.type()!=input.data.type():
                    self.alpha = self.alpha.type_as(input.data)
                self.alpha = Variable(self.alpha)

        target = target.view(-1,1) # N*H*W

        logpt = F.log_softmax(input) # N*H*W,C
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = nn.Softmax()(input) #torch.Size([192, 32])
        pt = pt.gather(1,target)
        if self.alpha is not None:
            at = self.alpha.gather(1,target)
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
if __name__ == '__main__':
    from loss import CrossEntropyLoss2d
    import numpy as np
    max_error  = 0
    seed=17*19
    np.random.seed()
    torch.initial_seed()
#cpu version:
    for i in range(1000):
        m = nn.Conv2d(16, 4, (3, 3))
        # input is of size N x C x height x width
        input = autograd.Variable(torch.randn(3, 16, 10, 10))
        # each element in target has to have 0 <= value < C
        target = autograd.Variable(torch.LongTensor(3, 8, 8).random_(0, 4))
        loss = FocalLoss(gamma=0,alpha=1)
        output1 = loss(m(input), target)


        loss = CrossEntropyLoss2d()
        output2 = loss(m(input), target)
        if abs(output2.data[0]-output1.data[0])>max_error:  max_error = abs(output2.data[0]-output1.data[0])
    print(max_error)
    max_error  = 0
#GPU version:
    torch.cuda.seed_all()
    for i in range(1000):
        m = nn.Conv2d(16, 4, (3, 3)).float()
        m.cuda()
        # input is of size N x C x height x width
        input = autograd.Variable(torch.randn(3, 16, 10, 10).cuda())
        # each element in target has to have 0 <= value < C
        target = autograd.Variable(torch.LongTensor(3, 8, 8).random_(0, 4).cuda())
        loss = FocalLoss(gamma=0,alpha=1)
        output1 = loss(m(input), target)
        loss = CrossEntropyLoss2d()
        output2 = loss(m(input), target)
        if abs(output2.data[0]-output1.data[0])>max_error:  max_error = abs(output2.data[0]-output1.data[0])
    print(max_error)

