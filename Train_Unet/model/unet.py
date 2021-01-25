import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader


def layer(d_in, d_out, k, p):
    '''
    a function that returns a Sequential of conv, BN and Lrelu
    :param d_in: input dim
    :param d_out: output dim
    :param k: kernel size
    :param p: padding
    :return: nn sequential
    '''
    conv = nn.Conv2d(d_in, d_out, kernel_size=k, padding=p)
    return nn.Sequential(conv, nn.BatchNorm2d(d_out), nn.LeakyReLU(inplace=False))


def conv_block(d_in, d_out, out_dim=8,out=False,k=3, p=1):
    '''
    a function that creates triple layers
    :param d_in: input dim
    :param d_out: output dim
    :param k: kernel size
    :param p: padding
    :return: 3 conv layers in a nn Sequential module
    '''
    block = []
    block.append(layer(d_in, d_out, k, p))
    for _ in range(2):
        block.append(layer(d_out, d_out, k, p))
    return nn.Sequential(*block)


class enc(nn.Module):
    '''
    encoder portion of unet
    '''
    def __init__(self, dummy=None, disc=False):
        super(enc, self).__init__()
        self.disc = disc
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        if self.disc:
            self.lev1 = conv_block(9, 16)
            self.lev2 = conv_block(16,32)
            self.lev3 = conv_block(32,32)
            self.lev4 = conv_block(32,64)
            self.lev5 = conv_block(64,64)
        else:
            #question 1 and question 2 unet architectures
            self.lev1 = conv_block(1,16)
            self.lev2 = conv_block(16,32)
            self.lev3 = conv_block(32,32)
            self.lev4 = conv_block(32,64)
            self.lev5 = conv_block(64,64)


    def forward(self,x):
        x1 = self.lev1(x)
        x1p = self.pool(x1)
        x2 = self.lev2(x1p)
        x2p = self.pool(x2)
        x3 = self.lev3(x2p)
        x3p = self.pool(x3)
        x4 = self.lev4(x3p)
        x4p = self.pool(x4)
        x5 = self.lev5(x4p)
        return (x5,x4,x3,x2,x1) if not self.disc else x5



class dec(nn.Module):
    '''
    decoder portion of unet
    '''
    def __init__(self, x):
        super(dec, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.inp = x[0].shape[1]
        self.d4 = self.inp + x[1].shape[1]
        self.d3 = self.d4 + x[2].shape[1]
        self.d2 = self.d3 + x[3].shape[1]
        self.d1 = int(self.d2/2) + x[4].shape[1]
        self.lev5 = conv_block(self.inp,self.inp)
        self.lev4 = conv_block(self.d4,self.d4)
        self.lev3 = conv_block(self.d3,self.d3)
        self.lev2 = conv_block(self.d2,int(self.d2/2))
        self.lev1 = conv_block(self.d1, int(self.d1/2))

    def forward(self,x):
        x1 = self.lev5(x[0])
        x1u = self.up(x1)
        x2 = self.lev4(torch.cat((x1u,x[1]),axis=1))
        x2u = self.up(x2)
        x3 = self.lev3(torch.cat((x2u,x[2]),axis=1))
        x3u = self.up(x3)
        x4 = self.lev2(torch.cat((x3u,x[3]),axis=1))
        x4u = self.up(x4)
        x5 = self.lev1(torch.cat((x4u,x[4]),axis=1))
        return x5



class unet(nn.Module):
    '''
    the unet class module
    '''
    def __init__(self, dummy, num_out=5):
        super(unet, self).__init__()
        self.encoder = enc()
        out = self.encoder(dummy)
        self.decoder = dec(out)
        self.calc_output_conv(dummy)
        self.out_layer = nn.Conv2d(self.output_dim[1],num_out, kernel_size=1, padding=0, stride=1)


    def forward(self,x):
        outs = self.encoder(x)
        o = self.decoder(outs)
        return self.out_layer(o)


    def calc_output_conv(self, dummy):
        self.output_dim = self.decoder(self.encoder(dummy)).shape


if __name__ == '__main__':
    from Train_Unet.data_prep.data_prep import load_and_save
    from Train_Unet.data_prep.make_dataset import segdataset
    s = r"C:\Users\jonat\Documents\Admin\Applications\Circle\data"
    p = ["ct_train"]
    n, classes = load_and_save(s, p, norm='per_patient')
    testing_dataset = segdataset(os.path.join(s, p[0]), (n[0][0], n[1][0]), p[0])
    testing_dataloader = DataLoader(testing_dataset)
    dummy = iter(testing_dataloader).next()[0]
    print("Image shape is ", dummy.shape)
    test_block = unet(dummy, num_out=len(classes))
    print("Network instantiated")
    print(test_block)
    print(test_block(dummy).shape)