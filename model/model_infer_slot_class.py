import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
from model.model_set_prediction import MLPClassifier
from model.model_set_prediction import HungarianHuberLoss,hungarian_huber_loss_gpt
from working_dir_root import learningR,learningR_res,SAM_pretrain_root,Load_feature,Weight_decay,Evaluation,Display_student,Display_final_SAM
from working_dir_root import Load_prototype_label
# from working_dir_root import Enable_teacher
from dataset.dataset import class_weights,Obj_num
import numpy as np
from image_operator import basic_operator   
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from SAM.segment_anything import  SamPredictor, sam_model_registry
from working_dir_root import Enable_student,Random_mask_temporal_feature,Random_mask_patch_feature,Display_fuse_TC_ST,Batch_size
from working_dir_root import Use_max_error_rejection,min_lr
from model import model_operator
# from MobileSAM.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from dataset.dataset import label_mask,Mask_out_partial_label
import random
from torch.optim import lr_scheduler

if Evaluation == True:
    learningR=0
    Weight_decay=0
# learningR = 0.0001
def select_gpus(gpu_selection):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print("Number of GPUs available:", num_gpus)
        if gpu_selection == "all":
            device = torch.device("cuda" if num_gpus > 0 else "cpu")
            # if num_gpus > 1:
            #     device = torch.device("cuda:0," + ",".join([str(i) for i in range(1, num_gpus)]))
        elif gpu_selection.isdigit():
            gpu_index = int(gpu_selection)
            device = torch.device("cuda:" + gpu_selection if gpu_index < num_gpus else "cpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device
class _Model_infer(object):
    def __init__(self, GPU_mode =True,num_gpus=1,Enable_teacher=True,Using_spatial_conv=True,Student_be_teacher=False,gpu_selection = "all",pooling="rank",TPC=True):
        super(_Model_infer, self).__init__()
        
        if GPU_mode ==True:
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device = select_gpus(gpu_selection)
        else:
            device = torch.device("cpu")
        self.device = device
       
        self.inter_bz =100*Batch_size
        
        self.TPC = TPC
        
        # model_type = "vit_t"
        # sam_checkpoint = "./MobileSAM/weights/mobile_sam.pt"
        
      
        # self.predictor = SamPredictor(self.sam) 
         
         
        self.MLP_classifier =  MLPClassifier(feature_dim=128, category_number=Obj_num+1, hidden_dim=1024)
        # self.input_size = 128
        # resnet18 = models.resnet34(pretrained=True)
        # self.gradcam = None
        # self.Enable_teacher = Enable_teacher
        # Remove the fully connected layers at the end
        # partial = nn.Sequential(*list(resnet18.children())[0:-3])
        
        # # Modify the last layer to produce the desired feature map size
        # self.resnet = nn.Sequential(
        #     partial,
        #     nn.ReLU()
        # )
        # # if GPU_mode ==True:
        # #     self.VideoNets.cuda()
        
        # if GPU_mode == True:
        #     if num_gpus > 1 and gpu_selection == "all":
        #         # self.VideoNets.classifier = torch.nn.DataParallel(self.VideoNets.classifier)
        #         # self.VideoNets.blocks = torch.nn.DataParallel(self.VideoNets.blocks)
        #         self.VideoNets = torch.nn.DataParallel(self.VideoNets)
        #         self.VideoNets_S = torch.nn.DataParallel(self.VideoNets_S)


        #         self.resnet  = torch.nn.DataParallel(self.resnet )
        #         self.Vit_encoder   = torch.nn.DataParallel(self.Vit_encoder  )
        #         self.sam_model  = torch.nn.DataParallel(self.sam_model )
        self.MLP_classifier.to(device)
        # self.VideoNets_S.to(device)


        # self.VideoNets.classifier.to(device)
        # self.VideoNets.blocks.to(device)


        # self.resnet .to(device)
        
        if Evaluation:
            pass
            #  self.VideoNets.eval()
            #  self.VideoNets_S.eval()
        else:
            self.MLP_classifier.train(True)
        


         
        self.custome_MSE= torch.nn.MSELoss().to(device)
        self.custome_loss =  HungarianHuberLoss()
       
        # ], weight_decay=0.1)
        self.optimizer = torch.optim.Adam([
            # {'params': self.resnet.parameters(),'lr': learningR*0.1},
        {'params': self.MLP_classifier.parameters(),'lr': learningR},
        # {'params': self.VideoNets.blocks.parameters(),'lr': learningR*0.9}
        ], weight_decay=Weight_decay)
        # self.optimizer = torch.optim.AdamW ([
       
        #     if num_gpus > 1:
        #         self.optimizer = torch.nn.DataParallel(optself.optimizerimizer)
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 10, eta_min=min_lr, last_epoch=-1)  # Optional parameters explained below
        # self.scheduler_s = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer_s, 10, eta_min=min_lr, last_epoch=-1)  # Optional parameters explained below

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
    def forward(self,prototype,prototype_l,prototype_r):
        # self.res_f = self.resnet(input)
      
        activationLU = nn.ReLU()

 
        self.final_output = self.MLP_classifier(prototype)
        self.final_output_l = self.MLP_classifier(prototype_l)
        self.final_output_r = self.MLP_classifier(prototype_r)

        self.direct_frame_output = None
    
    def optimization(self, label,ordered_label):
        # for param_group in  self.optimizer.param_groups:
        #     param_group['lr'] = lr 
        self.optimizer.zero_grad()
        if Load_prototype_label == False:
            B, S_l, C= self.final_output_l.size()
            B, S_r, C= self.final_output_r.size()
            apending_l= torch.zeros((1,max(S_l-2,1),15))
            apending_r= torch.zeros((1,max(S_r-2,1),15))
            apending_l[:,:,14] =1
            apending_l= apending_l.to(self.device)
            apending_r[:,:,14] =1
            apending_r= apending_r.to(self.device)
            label_l =torch.cat([label[:,0:2,:],apending_l],dim=1)
            label_r =torch.cat([label[:,2:4,:],apending_r],dim=1)
            self.loss_l = self.custome_loss(self.final_output_l,label_l)
            self.loss_r = self.custome_loss(self.final_output_r,label_r)

        # self.loss_l = hungarian_huber_loss_gpt(self.final_output_l,label_l)
        # self.loss_r = hungarian_huber_loss_gpt(self.final_output_r,label_r)
        # self.loss_l = self.custome_loss(self.final_output_l,label[:,0:2,:])
        # self.loss_r = self.custome_loss(self.final_output_r,label[:,2:4,:])


            self.loss= self.loss_l + self.loss_r
        # self.lossEa.backward(retain_graph=True)
        else:
            self.loss =  self.custome_MSE (self.final_output , ordered_label)
        self.loss.backward( )

        self.optimizer.step()
        self.lossDisplay = self.loss. data.mean()
        # self.set_requires_grad(self.resnet, True)

 

    def optimization_slicevalid(self):

        pass