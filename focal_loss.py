import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
# Reference Url: https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
class FocalLoss(nn.Module):
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
            if isinstance(self.alpha,list): 
                assert len(self.alpha)==C,"Make sure len(self.alpha) compatible with C"+str(C)
                list_tmp = []
                for n in range(N):
                    list_tmp.append(self.alpha)
                self.alpha = torch.Tensor(list_tmp)

        target = target.view(-1,1) # N*H*W

        logpt = F.log_softmax(input) # N*H*W,C
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = nn.Softmax()(input) #torch.Size([192, 32])
        pt = pt.gather(1,target)
        if self.alpha is not None:
            at = Variable(self.alpha).gather(1,target)
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
if __name__ == '__main__':

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
    loss = FocalLoss(gamma=2)
    output = loss(m(input), target)
    print(output)
    output.backward()

    from loss import CrossEntropyLoss2d
    loss = CrossEntropyLoss2d()
    output = loss(m(input), target)
    print(output)
    output.backward()

