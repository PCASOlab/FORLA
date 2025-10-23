import torch
import torch.nn as nn

def build_3dconv_block(indepth, outdepth, k, s, p, Drop_out = False, final=False):
    if final == False:
        module = nn.Sequential(
            # nn.ReflectionPad2d((p[1],p[1],p[0],p[0])),
            # nn.Conv2d(indepth, outdepth,k, s, (0,0), bias=False),
            nn.Conv3d(indepth, outdepth, k, s, p, bias=False),

            # nn.InstanceNorm3d(outdepth),
            # nn.LayerNorm(outdepth),

            nn.BatchNorm3d(outdepth),
            # nn.GroupNorm(4*int(outdepth/basic_feature),outdepth),

            nn.ReLU(),
            # nn.ReLU(),
            # nn.Dropout(0.1)
        )
        if Drop_out == True:
            module = nn.Sequential(
                # nn.ReflectionPad2d((p[1],p[1],p[0],p[0])),
                # nn.Conv2d(indepth, outdepth,k, s, (0,0), bias=False),
                nn.Conv3d(indepth, outdepth, k, s, p, bias=False),

                # nn.LayerNorm(outdepth),
                nn.BatchNorm3d(outdepth),

                # nn.GroupNorm(4*int(outdepth/basic_feature),outdepth),

                nn.ReLU(),
                # nn.ReLU(),
                nn.Dropout(0.2)
            )

    else:
        module = nn.Sequential(
            # nn.ReflectionPad2d((p[1],p[1],p[0],p[0])),
            # nn.Conv2d(indepth, outdepth,k, s, (0,0), bias=False),
            nn.Conv3d(indepth, outdepth, k, s, p, bias=False),
            # nn.Tanh()
            # nn.LeakyReLU(0.1, inplace=True)
        )
    return module
class Conv3DBlock_layernorm(nn.Module):
    def __init__(self, indepth, outdepth, k, s, p, Drop_out=False, final=False):
        super().__init__()
        self.final = final
        self.conv = nn.Conv3d(indepth, outdepth, k, s, p, bias=False)
        self.relu = nn.GELU()

        if  final:
            self.drop = nn.Dropout(0.2) if Drop_out else None
            # LayerNorm only at the block output
            self.norm = nn.LayerNorm(outdepth)
        else:
            self.norm = None
            self.drop = nn.Dropout(0.2) if Drop_out else None


    def forward(self, x):
        x = self.conv(x)

        if  self.final:
            N, C, D, H, W = x.shape
            # Move channel to last for LayerNorm
            # x = x.permute(0, 2, 3, 4, 1).contiguous()   # (N, D, H, W, C)
            # x = self.norm(x)                            # normalize over channels
            # x = x.permute(0, 4, 1, 2, 3).contiguous()   # back to (N, C, D, H, W)
            if self.drop:
                x = self.drop(x)
            # x = self.relu(x)
        else:
            if self.drop:
                x = self.drop(x)
            x = self.relu(x)
            
        return x

def build_2dconv_block(indepth, outdepth, k, s, p, Drop_out = False, final=False):
    if final == False:
        module = nn.Sequential(
            # nn.ReflectionPad2d((p[1],p[1],p[0],p[0])),
            # nn.Conv2d(indepth, outdepth,k, s, (0,0), bias=False),
            nn.Conv2d(indepth, outdepth, k, s, p, bias=False),

            nn.BatchNorm2d(outdepth),
            # nn.GroupNorm(4*int(outdepth/basic_feature),outdepth),

            nn.GELU(),
            # nn.ReLU(),
            # nn.Dropout(0.1)
        )
        if Drop_out == True:
            module = nn.Sequential(
                # nn.ReflectionPad2d((p[1],p[1],p[0],p[0])),
                # nn.Conv2d(indepth, outdepth,k, s, (0,0), bias=False),
                nn.Conv2d(indepth, outdepth, k, s, p, bias=False),

                nn.BatchNorm2d(outdepth),
                # nn.GroupNorm(4*int(outdepth/basic_feature),outdepth),

                nn.GELU(),
                # nn.ReLU(),
                nn.Dropout(0.1)
            )

    else:
        module = nn.Sequential(
            # nn.ReflectionPad2d((p[1],p[1],p[0],p[0])),
            # nn.Conv2d(indepth, outdepth,k, s, (0,0), bias=False),
            nn.Conv2d(indepth, outdepth, k, s, p, bias=False),
            # nn.Tanh()
            # nn.LeakyReLU(0.1, inplace=True)
        )
    return module
 

class conv_devide_H(nn.Module): # devide the H by half and keep the D and W
    def __init__ (self, indepth,outdepth,k=(1,4,3),s=(1,2,1),p=(0,1,1)):
        super(conv_devide_H, self).__init__()
        self.conv_block =  Conv3DBlock_layernorm (indepth,outdepth,k,s,p)

    def forward(self, x):
        #"""Forward function (with skip connections)"""
        out =  self.conv_block(x)  # add skip connections

        # this is a self desined residual block for Deeper nets

        #local_bz,channel,H,W = out.size()
        #downsample = nn.AdaptiveAvgPool2d((H,W))(x)
        #_,channel2,_,_ = downsample.size()
        #out[:,0:channel2,:,:] = out[:,0:channel2,:,:]+  downsample
        return out


class conv_keep_all(nn.Module):
    def __init__(self, indepth, outdepth, k=(1,3, 3), s=(1,1, 1), p=(0,1, 1), resnet=False, final=False,dropout=False):
        super(conv_keep_all, self).__init__()
        self.conv_block = Conv3DBlock_layernorm(indepth, outdepth, k, s, p, dropout,final)
        self.resnet = resnet

    def forward(self, x):
        # """Forward function (with skip connections)"""
        # out = x+ self.conv_block(x)  # add skip connections
        if self.resnet == False:
            out = self.conv_block(x)  # add skip connections
        else:
            out = x + self.conv_block(x)
        return out

class conv_keep_all_true3D(nn.Module):
    def __init__(self, indepth, outdepth, k=(3,3, 3), s=(1,1, 1), p=(1,1, 1), resnet=False, final=False,dropout=False):
        super(conv_keep_all_true3D, self).__init__()
        self.conv_block = build_3dconv_block(indepth, outdepth, k, s, p, dropout,final)
        self.resnet = resnet

    def forward(self, x):
        # """Forward function (with skip connections)"""
        # out = x+ self.conv_block(x)  # add skip connections
        if self.resnet == False:
            out = self.conv_block(x)  # add skip connections
        else:
            out = x + self.conv_block(x)
        return out


class conv_dv_WH(nn.Module): # devide H and W keep the D
    def __init__(self, indepth, outdepth, k=(1,4, 4), s=(1,2, 2), p=(0,1, 1),dropout=False):
        super(conv_dv_WH, self).__init__()
        self.conv_block = build_3dconv_block(indepth, outdepth, k, s, p,dropout)

    def forward(self, x):
        # """Forward function (with skip connections)"""

        out = self.conv_block(x)  # add skip connections
        # local_bz,channel,H,W = out.size()
        # downsample = nn.AdaptiveAvgPool2d((H,W))(x)
        # _,channel2,_,_ = downsample.size()
        # out[:,0:channel2,:,:] = out[:,0:channel2,:,:]+  downsample
        return out

   

class conv_keep_all2d(nn.Module):
    def __init__(self, indepth, outdepth, k=(3, 3), s=(1, 1), p=(1, 1), resnet=False, final=False,dropout=False):
        super(conv_keep_all2d, self).__init__()
        self.conv_block = build_2dconv_block(indepth, outdepth, k, s, p, dropout,final)
        self.resnet = resnet

    def forward(self, x):
        # """Forward function (with skip connections)"""
        # out = x+ self.conv_block(x)  # add skip connections
        if self.resnet == False:
            out = self.conv_block(x)  # add skip connections
        else:
            out = x + self.conv_block(x)
        return out

class conv_dv_WH2d(nn.Module): # devide H and W keep the D
    def __init__(self, indepth, outdepth, k=(4, 4), s=(2, 2), p=(1, 1)):
        super(conv_dv_WH2d, self).__init__()
        self.conv_block = build_2dconv_block(indepth, outdepth, k, s, p)
    def forward(self, x):
        # """Forward function (with skip connections)"""

        out = self.conv_block(x)  # add skip connections
        # local_bz,channel,H,W = out.size()
        # downsample = nn.AdaptiveAvgPool2d((H,W))(x)
        # _,channel2,_,_ = downsample.size()
        # out[:,0:channel2,:,:] = out[:,0:channel2,:,:]+  downsample
        return out