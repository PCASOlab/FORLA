import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.model_2dcnn import _VideoCNN2d
from model. gradcam import GradCam
from working_dir_root import learningR, Call_gradcam
from dataset.dataset import class_weights
# learningR = 0.0001
# Call_gradcam = False 

class _Model_infer(object):
    def __init__(self, GPU_mode =True,num_gpus=1):
        self.VideoNets = _VideoCNN2d()
        resnet18 = models.resnet18(pretrained=True)
        
        # Remove the fully connected layers at the end
        partial = nn.Sequential(*list(resnet18.children())[0:-4])
        self.gradcam = None
        self.resnet = partial
        # Modify the last layer to produce the desired feature map size
        # self.resnet = nn.Sequential(
        #     partial,
        #     nn.ReLU()
        # )
        # if GPU_mode ==True:
        #     self.VideoNets.cuda()
        if GPU_mode ==True:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        else:
            device = torch.device("cpu")
        if GPU_mode == True:
            if num_gpus > 1:
                self.VideoNets = torch.nn.DataParallel(self.VideoNets)
                self.resnet  = torch.nn.DataParallel(self.resnet )
        self.VideoNets.to(device)
        self.resnet .to(device)
        # self.customeBCE = torch.nn.BCEWithLogitsLoss().to(device)
        weight_tensor = torch.tensor(class_weights, dtype=torch.float)

        self.customeBCE = torch.nn.BCELoss(weight=weight_tensor).to(device)

        # self.customeBCE = torch.nn.BCELoss().to(device)

        self.optimizer = torch.optim.Adam([
          {'params': self.resnet.parameters()},
            {'params': self.VideoNets .parameters()}
        ], lr=learningR )
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
    def forward(self,input3d):
        bz, ch, D, H, W = input3d.size()

        input = input3d[:,:,1,:,:] # first images
        self.res_f = self.resnet(input)
        self.output,  self. cam3D= self.VideoNets(self.res_f)

        if Call_gradcam == True:
            target_layer =self.VideoNets.blocks[-1].conv_block[0]

    # Create a Grad-CAM instance
            Gradcam = GradCam(self.VideoNets, target_layer)  
            # activations = register_hooks(self.VideoNets)
            # Get the model prediction
            target_class = 0  # Replace with the target class index
            self.gradcam  = Gradcam.generate(self.res_f, target_class)

            # with torch.no_grad():
            #     output,_ = self.VideoNets(self.res_f)

            # # Get the predicted class index
            # # pred_idx = torch.argmax(output).item()

            # # Get the Grad-CAM heatmap for the predicted class
            # # heatmap = gradcam.get_gradcam(pred_idx)
            # self.gradcam = heatmap
            # Normalize the heatmap
            # heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-5)

            # # Resize the heatmap to the original image size
            # heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))

            # # Apply colormap to the heatmap
            # heatmap_colormap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

            # # Superimpose the heatmap on the original image
            # result = cv2.addWeighted(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), 0.7, heatmap_colormap, 0.3, 0)

            # # Display the result
            # cv2.imshow('Grad-CAM', result)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()    



    def optimization(self, label):
        self.optimizer.zero_grad()
        self.set_requires_grad(self.VideoNets, True)
        self.set_requires_grad(self.resnet, True)



        self.loss=  self.customeBCE(self.output.view(label.size(0), -1), label)
        # self.lossEa.backward(retain_graph=True)
        self.loss.backward()

        self.optimizer.step()
        self.lossDisplay = self.loss. data.mean()