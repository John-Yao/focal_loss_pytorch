# focal_loss_pytorch
python version: python3.6   
os version: Ubuntu 16.04 Kernel 4.4.0-81-generic   
Objective：focal_loss.py is created for caculate pixel-wise  focal loss    
Attention: the code hasn't been used to train network. Maybe there are some bugs!      
Paper:https://arxiv.org/pdf/1708.02002.pdf    
Thanks to https://github.com/clcarwin/focal_loss_pytorch   
How to test: run "python3 focal_loss.py" in shell   

ToDo:   
- [ ] figure out whether FocalLoss is numerically stable
- [x] use the code to train network   

Test Result:
- 20171124:  The code has been use for tranning network(gamma = 2).The result is about 92% accuracy in local dataset but when I use cross entropy loss the result is about 93% accuracy. So maybe the FocalLoss is unsuitabe for my dataset or the FocalLoss is something wrong.
And when I set gamma = 0 for FocalLoss,the trainning dosen't work!
