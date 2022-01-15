import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
A (n, r) covaraite tensor S as input: (batchsize=n, num_cov=r)

K_l = W_0 + W_1 * S_l1 + W_2 * S_l2 + ... + S_lr
where S_l is the l-th row of S. it consists of `r` covariates

we initiate one kernel weight matrix for each covariate, and 
assume S_l1, S_l2, ... are all binary (1 or 0)

Compute kernel: For each minibatch of size n, with the kernel param W0 and W1, 
first convolve each data point in the minibatch with either W_0 or W_0+W_1 (depend on the covariate), 
then you concat all the output of n convolution, do batchnorm

The goal is that given no covariate (or all 0's for covariates), this is just a normal 2d convolution layer
But with the presence of some covariates, the kernel weight matrix changes because of the covariate, 
and that difference might lead to positive impact on the model performance.
"""

#########################
# The hybrid Conv2d layer
#########################

class Hybrid_Conv2d(nn.Module):
    """    
    (self, channel_in, channel_out, kernel_size, cov, stride=1, padding=0)
    kernel_size are 4d weights: (out_channel, in_channel, height, width)
    """    
    def __init__(self, channel_in, channel_out, kernel_size, num_cov, stride=1, padding=0):
        super(Hybrid_Conv2d, self).__init__()
        self.kernel_size = kernel_size # 4D weight (out_channel, in_channel, height, width)
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding
        self.num_cov = num_cov # number of covariates

        self.W_0 = nn.Parameter(torch.randn(kernel_size), requires_grad=True)
        self.W = []
        for r in range(self.num_cov):
            W_r = nn.Parameter(torch.randn(kernel_size), requires_grad=True)
            self.W.append(W_r)        
        
        self._initialize_weights()
        
    # weight initialization
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.W_0, mode='fan_out', nonlinearity='relu')
        for r in range(self.num_cov):
            nn.init.kaiming_normal_(self.W[r], mode='fan_out', nonlinearity='relu')
 
    def forward(self, x, cov):
        # input x is of shape = (batchsize, channel=3, width, height) e.g. (32, 3, 224, 224)
        # cov: 2d tensor of shape (batchsize, r): 
        # r = number of covariates per image; 
        # bs = batchsize = number of images
        
        outputs = []
        for i in range(cov.shape[0]): # for every image x[i]
            res = torch.zeros_like(self.W_0)
            for j in range(cov.shape[1]): # for j-th cov of x[i]
                # element-wise multiply the W_j * cov_ij then take sum of these products across the covariates of that one image
                res = res + ( torch.mul(self.W[j], cov[i][j]) ).to('cuda:0') # cov[i] is an array with shape (r,); cov[i][j] is either 1 or 0
            # kernel is the linear combination of all those weight matrices.
            # note W_0 would be the ONLY real kernel weight matrix iff all covariates are 0
            kernel = self.W_0 + res
            # now compute the convolution for each image in the batch, but using the hybrid kernel
            x_i = torch.unsqueeze(x[i], 0) # (3, 224, 224) -> (1, 3, 224, 224) for 4d weight shape matching
            out = F.conv2d(x_i, kernel, stride=self.stride, padding=self.padding)
            outputs.append(out) 
            # concat the outputs and return that
        outputs = torch.cat(outputs)
        return outputs