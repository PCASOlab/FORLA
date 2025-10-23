import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.model_3dcnn_linear import _VideoCNN
from working_dir_root import learningR,learningR_res,Evaluation,Display_final_SAM
from dataset.dataset import class_weights
from model import model_operator
import model.display. model_vis as modelVis

# learningR = 0.0001
def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
class _Model_infer(object):
    def __init__(self, GPU_mode =True,num_gpus=1):
        self.VideoNets = _VideoCNN(inputC=256)
        self.input_size = 128 
        # resnet18 = models.resnet34(pretrained=True)
        self.gradcam = None
        
        # Remove the fully connected layers at the end
        # partial = nn.Sequential(*list(resnet18.children())[0:-3])
        vgg = models.vgg16_bn(pretrained=True)
        vgg_trunk = nn.Sequential(*list(vgg.features.children())[:16])  # up to and including pool4 -> stride=16, C=512
        partial = vgg_trunk
        # Project C=512 -> C=384 to match VideoNets(inputC=384)
        # self.backbone = nn.Sequential(
        #     vgg_trunk,
        #     nn.Conv2d(512, 384, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU(inplace=True),
        # )
        # Modify the last layer to produce the desired feature map size
        self.resnet = nn.Sequential(
            partial,
            nn.ReLU()
        )
        # if GPU_mode ==True:
        #     self.VideoNets.cuda()
        if GPU_mode ==True:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        else:
            device = torch.device("cpu")
        # if GPU_mode == True:
        #     if num_gpus > 1:
        #         self.VideoNets = torch.nn.DataParallel(self.VideoNets)
        #         self.resnet  = torch.nn.DataParallel(self.resnet )
        self.VideoNets.to(device)
        self.resnet .to(device)
        if Evaluation:
            self.resnet.eval()
            self.VideoNets.eval()

        else:
            self.resnet.train(True)
            self.VideoNets.train(True)
            # self.resnet.apply(initialize_weights)
            self.VideoNets.apply(initialize_weights)


        
        weight_tensor = torch.tensor(class_weights, dtype=torch.float)
        self.customeBCE = torch.nn.MSELoss().to(device)
        # self.customeBCE = torch.nn.BCEWithLogitsLoss().to(device)

        # self.customeBCE = torch.nn.BCELoss(weight=weight_tensor).to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self. resnet.parameters(),'lr': learningR},
            {'params': self.VideoNets .parameters(),'lr': learningR}
        ] )
        # if GPU_mode ==True:
        #     if num_gpus > 1:
        #         self.optimizer = torch.nn.DataParallel(optself.optimizerimizer)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def forward(self,input,input_flows):
        # self.res_f = self.resnet(input)
        bz, ch, D, H, W = input.size()

        self.input_resample =   F.interpolate(input,  size=(D, self.input_size, self.input_size), mode='trilinear', align_corners=False)
        # self.
        flattened_tensor = self.input_resample.permute(0,2,1,3,4)
        flattened_tensor = flattened_tensor.reshape(bz * D, ch, self.input_size, self.input_size)
        flattened_tensor = self.resnet((flattened_tensor-128.0)/60.0)
        new_bz, new_ch, new_H, new_W = flattened_tensor.size()
        self.f = flattened_tensor.reshape (bz,D,new_ch,new_H, new_W).permute(0,2,1,3,4)
        # modelVis.plot_patch_pca_per_image(input,self.f.reshape (bz,D,new_ch,new_H* new_W).permute(0,1,3,2), title="backbone_features")
        self.output, self.slice_valid, self. cam3D= self.VideoNets(self.f,input_flows)

        # max_p,_ =  torch.max(self.c_logits,dim=2)
        # self.output =max_p.reshape(bz, class_num,1,1,1)
        self.raw_cam = self.cam3D
        if Display_final_SAM:
            with torch.no_grad():
                activationLU = nn.ReLU()

                post_processed_masks=model_operator.Cam_mask_post_process(activationLU(self.cam3D), input,self.output)
                # self.sam_mask_prompt_decode(activationLU(self.cam3D),self.f,input)

                
                self.cam3D = post_processed_masks 
        with torch.no_grad():
            self.final_output = self.output.detach().clone()
            self.direct_frame_output = None
    def optimization(self, label):
        self.optimizer.zero_grad()
        self.set_requires_grad(self.VideoNets, True)
        self.set_requires_grad(self.resnet, True)

        self.loss=  self.customeBCE(self.output.view(label.size(0), -1), label)
        # self.lossEa.backward(retain_graph=True)
        self.loss.backward( )

        self.optimizer.step()
        self.lossDisplay = self.loss. data.mean()