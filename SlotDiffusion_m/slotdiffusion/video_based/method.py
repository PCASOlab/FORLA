import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import time
from tqdm import tqdm
from dataset.dataset import myDataloader
from working_dir_root import GPU_mode ,Continue_flag ,Visdom_flag ,Display_flag ,loadmodel_index  ,img_size,Load_flow,Load_feature
from working_dir_root import Max_lr, learningR,learningR_res,Save_feature_OLG,sam_feature_OLG_dir, Evaluation,Save_sam_mask,output_folder_sam_masks
from working_dir_root import Enable_student,Batch_size,selected_data,Display_down_sample,Evaluation_slots,Data_percentage
from nerv.training import BaseMethod, CosineAnnealingWarmupRestarts
import math
import eval_slots
from working_dir_root import Output_root
from model import  model_experiement, model_infer
from display import Display
from eval import calculate_model_flops_slot_difussion
# Output_root = Output_root+ "Obj_centric_temp2MLP" + selected_data + "/"

from SlotDiffusion_m.slotdiffusion.video_based.models import to_rgb_from_tensor, get_lr, \
    cosine_anneal, make_one_hot
from SlotDiffusion_m.slotdiffusion.video_based.vis import torch_draw_rgb_mask


def build_method(**kwargs):
    params = kwargs['params']
    if params.model in ['SAVi', 'STEVE', 'SAViDiffusion']:
        return eval(f'{params.model}Method')(**kwargs)
    elif 'VAE' in params.model:
        return VAEMethod(**kwargs)
    else:
        raise NotImplementedError(f'{params.model} method is not implemented.')


class SlotBaseMethod(BaseMethod):
    """Base method in this project."""
    # def __init__(self):
    #     super().__init__( )
        
    #     # Add your custom attribute or logic here
    #     self.new_value = 0
    #     print(f"CustomMethod initialized with new_value={0}")
    # @property
    def set_my_data_loader(self):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = device
        self.my_data_loader = myDataloader(img_size = img_size,Display_loading_video = False,Read_from_pkl= True,Save_pkl = False,Load_flow=Load_flow, Load_feature=Load_feature,Train_list='else',Device=self.device)
        self.model_infer = model_infer._Model_infer()
        # self.out_root = Output_root + self.params.model
        dataset_tag = "+".join(selected_data) if isinstance(selected_data, list) else selected_data
        self.Output_root = Output_root+ self.params.model + dataset_tag +str(Data_percentage)+ "/"

        # return dataLoader
         
    @property
    def num_cls(self):
        # hard coded number of classes to visualize seg masks
        if 'movi' in self.params.dataset.lower():
            if self.params.movi_level.lower() == 'c':
                return 11  # 10 obj + 1 bg
            if self.params.movi_level.lower() in ['d', 'e']:
                return 20  # should be >=25, but we hack it with 20
        # we don't `-1` because bg also counts for a slot
        return self.params.slot_dict['num_slots']

    @property
    def vis_fps(self):
        # Physion
        if 'physion' in self.params.dataset.lower():
            return 16 // self.params.frame_offset
        # MOVi, etc.
        else:
            return 8

    def _convert_video(self, video, caption=None):
        """Convert torch.tensor video in [-1, 1] to wandb.Video."""
        # a batch of [T, 3, H, W] --> [T, 3, B*H, W]
        if isinstance(video, (list, tuple)):
            video = torch.cat(video, dim=2)
        video = to_rgb_from_tensor(video)
        video = (video * 255.).cpu().numpy().astype(np.uint8)
        return wandb.Video(video, fps=self.vis_fps, caption=caption)

    def _get_sample_idx(self, N, dst):
        """Load samples uniformly from the dataset."""
        dst_len = len(dst.files)
        if N > dst_len:
            N = dst_len
        N = N - 1 if dst_len % N != 0 else N
        sampled_idx = torch.arange(0, dst_len, dst_len // N).numpy()
        return sampled_idx

    @torch.no_grad()
    def validation_epoch(self, model, san_check_step=-1, sample_video=True):
        """Validate one epoch.
        We aggregate the avg of all statistics and only log once.
        """
        self._is_epoch_end = (san_check_step <= 0)
        self._is_last_epoch = (self.epoch == self.max_epochs)
        # super().validation_epoch(model, san_check_step=san_check_step)
        if self.local_rank != 0:
            return
        # visualization after every epoch
        vali_data_loader = myDataloader(img_size = img_size,Display_loading_video = False,Read_from_pkl= True,Save_pkl = False,Load_flow=Load_flow, Load_feature=Load_feature,Train_list='else',Device=self.device)
        input_videos, labels= vali_data_loader.read_a_batch()
        input_videos_GPU = torch.from_numpy(np.float32(input_videos))
        labels_GPU = torch.from_numpy(np.float32(labels))
        input_videos_GPU = input_videos_GPU.to (self.device)
        labels_GPU = labels_GPU.to (self.device)

        bz, ch, D, H, W = input_videos_GPU.size()
        activationLU = nn.ReLU()

        input_size = 128
        self.input_resample =   F.interpolate(input_videos_GPU,  size=(D, input_size, input_size), mode='trilinear', align_corners=False)
        self.input_resample= (self.input_resample-124.0)/60.0
        batch_data = {"img":  self.input_resample.permute(0,2,1,3,4),
                        "data_idx": torch.zeros(bz)}
        slot_num = self.params.slot_dict['num_slots']
        out_dict = model(batch_data)
        masks = out_dict['masks'] # [B,T, N, H, W]
        _,_,_, H_m, W_m = masks.size()
        masks = masks.reshape(bz,D,slot_num,H_m, W_m)
        hard_masks = make_one_hot(masks, dim=2).type_as(masks)

        # MODEL_infer=[]
        # self.model_infer.cam3D = 0
        # if Display_flag == True and read_id%Display_down_sample == 0:
        vali_data_loader.labels  = vali_data_loader.labels * 0 +1
        displayer = Display(GPU_mode)
        self.model_infer.cam3D =  hard_masks.permute(0,2,1,3,4)
        bz, ch_n, D, H, W = self.model_infer.cam3D.size()
        
        self.model_infer.raw_cam = self.model_infer.cam3D
        self.model_infer.final_output = torch.ones(bz,ch_n,1,1,1)
        self.model_infer.gradcam = None
        self.model_infer.direct_frame_output = None
        
        displayer.train_display(self.model_infer,vali_data_loader,0,self.Output_root)
        # slot_video = self._make_slots_video(video, None, masks)
        # results.append(slot_video)
        # hard seg_mask by taking one-hot
        # hard_masks = make_one_hot(masks, dim=1).type_as(masks)

        # if sample_video:
        #     self._sample_video(model)
    def validation_batch_data(self,batch_data,model,vali_data_loader,read_id):
        out_dict = model(batch_data)
        masks = out_dict['masks'] # [B,T, N, H, W]
        if len ( masks.size())==6:
             bz,D,slot_n,_, H_m, W_m =masks.size()
        # _,_,_, H_m, W_m = masks.size()
             masks = masks.reshape( bz,D,slot_n, H_m, W_m )
        hard_masks = make_one_hot(masks, dim=2).type_as(masks)

        # MODEL_infer=[]
        # self.model_infer.cam3D = 0
        # if Display_flag == True and read_id%Display_down_sample == 0:
        vali_data_loader.labels  = vali_data_loader.labels * 0 +1
        displayer = Display(GPU_mode)
        self.model_infer.cam3D =  hard_masks.permute(0,2,1,3,4)
        bz, ch_n, D, H, W = self.model_infer.cam3D.size()
        
        self.model_infer.raw_cam = self.model_infer.cam3D
        self.model_infer.final_output = torch.ones(bz,ch_n,1,1,1)
        self.model_infer.gradcam = None
        self.model_infer.direct_frame_output = None
        
        displayer.train_display(self.model_infer,vali_data_loader,read_id,self.Output_root)

    def _configure_optimizers(self):
        """Returns an optimizer, a scheduler and its frequency (step/epoch)."""
        optimizer = super()._configure_optimizers()[0]

        lr = self.params.lr
        total_steps = self.params.max_epochs * len(self.train_loader)
        warmup_steps = self.params.get('warmup_steps_pct', 0.) * total_steps

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            total_steps,
            max_lr=lr,
            min_lr=lr / 100.,
            warmup_steps=warmup_steps,
        )

        return optimizer, (scheduler, 'step')
    def fit(self, resume_from='', san_check_val_step=2):
        """Train the model.

        Args:
            resume_from (str, optional): pre-trained weight path.
                Default: ''.
            san_check_val_step (int, optional): run a few val steps to verify
                the model, logging, checkpointing etc implementations.
                Default: 2.
        """
        # automatically detect existing checkpoints
        self.load_ckp(ckp_path=resume_from)
        self.set_my_data_loader()


        # run several val steps as sanity check
        if san_check_val_step > 0:
            self.validation_epoch(
                self.model.module, san_check_step=san_check_val_step)

        self._training_start()
        if Evaluation_slots:
            self.epoch = 0

        for _ in range(self.epoch, self.max_epochs):
            self.training_epoch()
        output_file = eval_slots.process_metrics_from_excel(self.Output_root + "/metrics_video.xlsx", self.Output_root)

        self._training_end()

    def _training_epoch_end(self):
        """Things to do at the end of every training epoch."""
        if self.scheduler_method == 'epoch':
            self.scheduler.step()
        self.epoch += 1
        self._is_last_epoch = (self.epoch == self.max_epochs)
        self._is_epoch_end = True
        self.stats_dict = None

        # call the same method for model
        self.model.module._training_epoch_end(method=self)

        self.save_ckp(save_loader=False)
        if self.use_ddp:
            torch.distributed.barrier()

        # run one epoch of validation after each training epoch
        # if (self.epoch + 1) % self.eval_interval == 0 or self._is_last_epoch:
        # metrics = self.validation_epoch(self.model.module)
        # self.save_ckp(save_loader=False, metrics=metrics)
        self.save_ckp(save_loader=False )

    def _training_step_end(self):
        """Things to do at the end of every training step."""
        if self.scheduler_method == 'step':
            self.scheduler.step()
        self.it += 1

        # call the same method for model
        self.model.module._training_step_end(method=self)

        if (self.epoch_it + 1) % self.save_iter == 0:
            self.save_ckp(save_loader=False)
            # sync DDP training processes at the end of step
            if self.use_ddp:
                torch.distributed.barrier()
    def _log_train(self, out_dict):
        """Log statistics in training to wandb."""
        # only log results for rank 0
        if self.local_rank != 0:
            return

        # accumulate results till print interval
        self._accumulate_stats(out_dict)

        if (self.epoch_it + 1) % self.print_iter != 0 and \
                (self.epoch_it + 1) != self.train_steps:
            return

        out_dict = {f'train/{k}': v.avg for k, v in self.stats_dict.items()}
        out_dict['train/epoch'] = self.epoch
        out_dict['train/it'] = self.it
        out_dict['train/lr'] = get_lr(self.optimizer)
        if self.use_fp16:
            out_dict['train/fp16_loss_scale'] = self.grad_scaler.get_scale()
        wandb.log(out_dict, step=self.it)
        self.stats_dict = None
    def training_epoch(self,my_data_loader=None):
        """Train one epoch with custom modifications."""
        self._training_epoch_start()
        self.set_my_data_loader()
        # Use iter dataloader in order to save its state
        # self.iter_train_loader = iter(self.my_data_loader.read_a_batch())
        # train_steps = (
        #     len(self.train_loader.sampler) -
        #     self.train_loader.sampler.real_counter(
        #         self.iter_train_loader)) // self.params.train_batch_size
        train_steps = math.ceil(self.my_data_loader.video_num / self.my_data_loader.batch_size)
        self.train_steps = train_steps
        tqdm_desc = f'Train epoch {self.epoch}, rank {self.local_rank}'
        with tqdm(total=train_steps+1, desc=tqdm_desc) as t:
            t1 = time.time()
            read_id=0
            while(1):
            # for batch_idx, batch_data in enumerate(self.iter_train_loader):
                # Custom modification: Log batch index
                print(f"Processing batch {read_id}")
                input_videos, labels= self.my_data_loader.read_a_batch(this_epoch= self.epoch)
                input_videos_GPU = torch.from_numpy(np.float32(input_videos))
                labels_GPU = torch.from_numpy(np.float32(labels))
                input_videos_GPU = input_videos_GPU.to (self.device)
                labels_GPU = labels_GPU.to (self.device)

                bz, ch, D, H, W = input_videos_GPU.size()
                activationLU = nn.ReLU()

                input_size = self.params.resolution[0]
                self.input_resample =   F.interpolate(input_videos_GPU,  size=(D, input_size, input_size), mode='trilinear', align_corners=False)
                self.input_resample= (self.input_resample-124.0)/60.0
                batch_data = {"img":  self.input_resample.permute(0,2,1,3,4),
                                "data_idx": torch.zeros(bz)}

                # Data time
                t2 = time.time()
                data_time = t2 - t1
                # Set the batch idx
                # self.epoch_it = batch_idx
                self.epoch_it = read_id
                

                # Model forward, loss computation, backward and optimize
                if Evaluation_slots == False:
                    batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                    self._training_step_start()
                    out_dict = self._training_step(batch_data)

                    # calculate_model_flops_slot_difussion()
                    # Custom modification: Add extra loss term if needed
                    # out_dict['loss'] += custom_loss_term(batch_data)
                    # Forward time
                    t1 = time.time()
                    forward_time = t1 - t2
                    out_dict['data_time'] = self._make_tensor(data_time)
                    out_dict['forward_time'] = self._make_tensor(forward_time)
                    print("forward_time: " + str(forward_time) + " data_time: " + str(data_time)) 
                    # Logging
                    self._log_train(out_dict)
                    self._training_step_end()
                    t.set_postfix(loss=f"{out_dict['loss'].item():.4f}")
                else:
                    self.validation_batch_data(batch_data,self.model.module,self.my_data_loader,read_id)

                
                t.update(1)
                read_id +=1
                if self.my_data_loader.all_read_flag ==1:
                    Save_feature_OLG = False
                    #remove this for none converting mode
                    # epoch +=1

                    # print("finished epoch" + str (epoch) )
                    self.my_data_loader.all_read_flag = 0
                    read_id=0
                    break

        self._training_epoch_end()

        # Custom modification: Print additional logging
        print("Epoch completed with custom logging")


class SAViMethod(SlotBaseMethod):
    """SAVi model training method."""

    def _make_video_grid(self, imgs, recon_combined, recons, masks):
        """Make a video of grid images showing slot decomposition."""
        # merge masks together, and map it to the video
        # [T, 3, H, W], [T, 3, H, W]
        rgb_mask, colored_mask = \
            torch_draw_rgb_mask(imgs, masks[:, :, 0].argmax(1))
        # combine images in a way so we can display all outputs in one grid
        # output rescaled to be between 0 and 1
        out = torch.cat(
            [
                imgs.unsqueeze(1),  # original images
                recon_combined.unsqueeze(1),  # reconstructions
                colored_mask.unsqueeze(1),  # colored masks
                rgb_mask.unsqueeze(1),  # rgb masks
                recons * masks + (1. - masks),  # each slot
            ],
            dim=1,
        ).cpu()  # [T, num_slots+2, 3, H, W]
        # stack the slot decomposition in all frames to a video
        out = torch.stack([
            vutils.make_grid(
                out[i],
                nrow=out.shape[1],
                pad_value=-1.,
            ) for i in range(recons.shape[0])
        ])  # [T, 3, H, (num_slots+2)*W]
        return out

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = self.val_loader.dataset
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        results = []
        for idx in sampled_idx:
            data_dict = dst.get_video(idx)
            video = data_dict['video'].float().to(model.device)
            in_dict = {'img': video[None]}
            out_dict = model(in_dict)
            out_dict = {k: v[0] for k, v in out_dict.items()}
            recon_img, recons, masks = \
                out_dict['recon_img'], out_dict['recons'], out_dict['masks']
            imgs = video.type_as(recon_img)
            slot_video = self._make_video_grid(imgs, recon_img, recons, masks)
            results.append(slot_video)
        wandb.log({'val/video': self._convert_video(results)}, step=self.it)
        torch.cuda.empty_cache()


class STEVEMethod(SlotBaseMethod):
    """STEVE model training method."""

    def _configure_optimizers(self):
        """Returns an optimizer, a scheduler and its frequency (step/epoch)."""
        assert self.params.optimizer.lower() == 'adam'
        assert self.params.weight_decay <= 0.
        lr = self.params.lr
        dec_lr = self.params.dec_lr

        # STEVE uses different lr for its Transformer decoder and other parts
        sa_params = list(
            filter(
                lambda kv: 'trans_decoder' not in kv[0] and kv[1].
                requires_grad, self.model.named_parameters()))
        dec_params = list(
            filter(lambda kv: 'trans_decoder' in kv[0],
                   self.model.named_parameters()))

        params_list = [
            {
                'params': [kv[1] for kv in sa_params],
            },
            {
                'params': [kv[1] for kv in dec_params],
                'lr': dec_lr,
            },
        ]

        optimizer = optim.Adam(params_list, lr=lr, weight_decay=0.)

        total_steps = self.params.max_epochs * len(self.train_loader)
        warmup_steps = self.params.warmup_steps_pct * total_steps

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            total_steps,
            max_lr=(lr, dec_lr),
            min_lr=0.,
            warmup_steps=warmup_steps,
        )

        return optimizer, (scheduler, 'step')

    @torch.no_grad()
    def validation_epoch(self, model, san_check_step=-1):
        """Validate one epoch.

        We aggregate the avg of all statistics and only log once.
        """
        # STEVE's Transformer-based decoder autoregressively reconstructs the
        # video, which is super slow
        # therefore, we only visualize scene decomposition results
        # but don't show the video reconstruction
        # change this if you want to see reconstruction anyways
        self.recon_video = False
        super().validation_epoch(model, san_check_step=san_check_step)

    @staticmethod
    def _make_video(video, other_videos):
        """videos are of shape [T, C, H, W]"""
        return VAEMethod._make_video(video, other_videos)

    @staticmethod
    def _make_slots_video(video, masks):
        """[T, 3, H, W], [T, N, H, W]"""
        return SAViDiffusionMethod._make_slots_video(video, None, masks)

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        model.testing = True  # we only want the slots & masks
        dst = self.val_loader.dataset

        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        results, gt_results, recon_results = [], [], []
        for idx in sampled_idx:
            data_dict = dst.get_video(idx)
            video = data_dict['video'].float().to(model.device)
            in_dict = {'img': video[None]}
            out_dict = model(in_dict)
            masks = out_dict['masks'][0]  # [T, num_slots, H, W]
            save_video = self._make_slots_video(video, masks)
            results.append(save_video)
            # in case GT masks are provided
            if 'masks' in data_dict:
                gt_masks = data_dict['masks'].to(model.device)  # [T, H, W]
                gt_masks = F.one_hot(gt_masks, num_classes=self.num_cls)
                # to [T, num_slots, H, W]
                gt_masks = gt_masks.permute(0, 3, 1, 2).float()
                gt_video = self._make_slots_video(video, gt_masks)
                gt_results.append(gt_video)

            # recon video is very slow (autoreg sampling patches)
            # at most do it at the end of each epoch
            # however, we will do this at the end of training
            if (not self.recon_video or not self._is_epoch_end) \
                    and not self._is_last_epoch:
                continue

            # reconstruct the video by autoregressively generating patch tokens
            # using Transformer decoder conditioned on slots
            slots = out_dict['slots'][0]  # [T, num_slots, slot_size]
            recon_video = model.recon_img(
                slots, bs=16, verbose=(idx == sampled_idx[0]))
            save_video = self._make_video(video, recon_video)
            recon_results.append(save_video)
            torch.cuda.empty_cache()

        log_dict = {'val/masked_video': self._convert_video(results)}
        if gt_results:
            log_dict['val/gt_masked_video'] = self._convert_video(gt_results)
        if self.recon_video:
            log_dict['val/recon_video'] = self._convert_video(recon_results)
        wandb.log(log_dict, step=self.it)
        torch.cuda.empty_cache()
        model.testing = False

    def _log_train(self, out_dict):
        """Log statistics in training to wandb."""
        super()._log_train(out_dict)

        if self.local_rank != 0 or (self.epoch_it + 1) % self.print_iter != 0:
            return

        # log all the lr
        log_dict = {
            'train/lr': get_lr(self.optimizer),
            'train/dec_lr': self.optimizer.param_groups[1]['lr'],
        }
        wandb.log(log_dict, step=self.it)


class SAViDiffusionMethod(SlotBaseMethod):
    """SlotAttention with Diffusion decoder model training method."""

    def _configure_optimizers(self):
        """Returns an optimizer, a scheduler and its frequency (step/epoch)."""
        if self.params.optimizer.lower() == 'adam':
            opt = optim.Adam
        elif self.params.optimizer.lower() == 'adamw':
            opt = optim.AdamW
        else:
            raise ValueError('Should use Adam or AdamW optimizer!')
        if self.params.weight_decay > 0.:
            assert self.params.optimizer.lower() == 'adamw', \
                'Should use AdamW optimizer for weight decay!'

        lr = self.params.lr
        dec_lr = self.params.get('dec_lr', lr)

        # may need different lr for the DM decoder and other parts
        sa_params = list(
            filter(
                lambda kv: 'dm_decoder' not in kv[0] and kv[1].requires_grad,
                self.model.named_parameters()))
        dec_params = list(
            filter(lambda kv: 'dm_decoder' in kv[0],
                   self.model.named_parameters()))

        params_list = [
            {
                'params': [kv[1] for kv in sa_params],
                'lr': lr,
                'weight_decay': 0.,
            },
            {
                'params': [kv[1] for kv in dec_params],
                'lr': dec_lr,
                'weight_decay': self.params.weight_decay,
            },
        ]

        optimizer = opt(params_list)

        total_steps = self.params.max_epochs * len(self.train_loader)
        warmup_steps = self.params.warmup_steps_pct * total_steps

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            total_steps,
            max_lr=(lr, dec_lr),
            min_lr=0.,
            warmup_steps=warmup_steps,
        )

        return optimizer, (scheduler, 'step')

    @staticmethod
    def _make_slots_video(video, recon_video, masks):
        """[T, 3, H, W]x2, [T, N, H, W]"""
        # merge masks together, and map it to the video
        # [T, 3, H, W], [T, 3, H, W]
        rgb_mask, colored_mask = torch_draw_rgb_mask(video, masks.argmax(1))
        # make per-slot masked video, [T, N, 3, H, W]
        masked_video = video.unsqueeze(1) * masks.unsqueeze(2) + \
            (1. - masks.unsqueeze(2))
        video_lst = [
            video.unsqueeze(1),  # [T, 1, 3, H, W]
            colored_mask.unsqueeze(1),  # [T, 1, 3, H, W]
            rgb_mask.unsqueeze(1),  # [T, 1, 3, H, W]
            masked_video,  # [T, N, 3, H, W]
        ]
        if recon_video is not None:
            video_lst.insert(1, recon_video.unsqueeze(1))  # [T, 1, 3, H, W]
        out = torch.cat(video_lst, dim=1).cpu()  # [T, N + 2, 3, H, W]
        # stack the slot decomposition in all frames to a video
        out = torch.stack([
            vutils.make_grid(
                out[i],
                nrow=out.shape[1],
                pad_value=-1.,
            ) for i in range(out.shape[0])
        ])  # [T, 3, H, (N+2)*W]
        return out

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        model.testing = True  # we only want the slots & masks
        dst = self.val_loader.dataset
        

        # 1. sample a batch of video clips, evaluate recon loss
        # slow, only do this at the end of each epoch OR at the end of training
        if self._is_epoch_end or self._is_last_epoch:
            sampled_idx = self._get_sample_idx(self.params.val_batch_size, dst)
            collate_fn = self.val_loader.collate_fn
            data_dict = collate_fn([dst[i] for i in sampled_idx])
            data_dict = {k: v.to(model.device) for k, v in data_dict.items()}
            dpm_log_dict = model.log_images(
                data_dict,
                ret_intermed=False,
                use_ddim=False,
                use_dpm=True,
                verbose=True,
            )  # DPM-Solver recon, the fastest, and not so bad quality
            imgs = data_dict['img']
            dpm_recon_loss = F.mse_loss(dpm_log_dict['samples'], imgs)
            wandb_dict = {
                'val/dpm_sample_video_recon_loss': dpm_recon_loss.cpu().item(),
            }
            print(f'val/dpm_recon_loss: {dpm_recon_loss.cpu().item():.4f}')
        else:
            wandb_dict = {}

        # 2. sample a few videos for visualization
        video_idx = self._get_sample_idx(self.params.n_samples, dst)
        results, hard_results, gt_results = [], [], []
        for idx in video_idx:
            data_dict = dst.get_video(idx)
            video = data_dict['video'].to(model.device)  # [T, 3, H, W]
            in_dict = {'img': video[None]}
            out_dict = model(in_dict)
            masks = out_dict['masks'][0]  # [T, N, H, W]
            slot_video = self._make_slots_video(video, None, masks)
            results.append(slot_video)
            # hard seg_mask by taking one-hot
            hard_masks = make_one_hot(masks, dim=1).type_as(masks)
            hard_slot_video = self._make_slots_video(video, None, hard_masks)
            hard_results.append(hard_slot_video)
            # in case GT masks are provided
            if 'masks' in data_dict:
                gt_masks = data_dict['masks'].to(model.device)  # [T, H, W]
                gt_masks = F.one_hot(gt_masks, num_classes=self.num_cls)
                # to [T, num_slots, H, W]
                gt_masks = gt_masks.permute(0, 3, 1, 2).float()
                gt_video = self._make_slots_video(video, None, gt_masks)
                gt_results.append(gt_video)
        wandb_dict['val/masked_video'] = self._convert_video(results)
        wandb_dict['val/hard_masked_video'] = \
            self._convert_video(hard_results)
        if gt_results:
            wandb_dict['val/gt_masked_video'] = self._convert_video(gt_results)
        wandb.log(wandb_dict, step=self.it)
        torch.cuda.empty_cache()

        # 3. reconstruct the sampled videos
        # slow, only do this at the end of each epoch and with DPM_Solver
        use_dpm = True
        if (not use_dpm or not self._is_epoch_end) and \
                not self._is_last_epoch:
            model.testing = False
            return
        results = {}
        for idx in video_idx:
            data_dict = dst.get_video(idx)
            video = data_dict['video'].to(model.device)  # [T, 3, H, W]

            # split the video into several clips to avoid OOM
            clip_len = self.params.val_batch_size * 4
            all_log_dict = {}
            for img_idx in range(0, video.shape[0], clip_len):
                in_dict = {'img': video[None, img_idx:img_idx + clip_len]}
                # same init noise for all frames to gain temporal consistency
                log_dict = model.log_images(
                    in_dict,
                    ret_intermed=True,
                    use_ddim=False,
                    use_dpm=True,
                    same_noise=True,
                    verbose=self._is_last_epoch or (idx == video_idx[0]),
                )
                log_dict = {k: v[0].cpu() for k, v in log_dict.items()}
                '''
                    - 'diffusion_row': from x0 (input imgs) to xT (noise)
                    - 'denoise_row': from xT denoises to x0
                    - 'samples': the final denoised (reconstructed) imgs
                    - 'masks': [clip_len, N, H, W]
                '''
                if not all_log_dict:
                    all_log_dict = {k: [v] for k, v in log_dict.items()}
                else:
                    for k, v in log_dict.items():
                        all_log_dict[k].append(v)
            log_dict = {k: torch.cat(v, 0) for k, v in all_log_dict.items()}
            masks, recon_video = log_dict.pop('masks'), log_dict.pop('samples')
            video = video.cpu()
            slot_video = self._make_slots_video(video, recon_video, masks)
            log_dict['video'] = slot_video
            if not results:
                results = {k: [v] for k, v in log_dict.items()}
            else:
                for k, v in log_dict.items():
                    results[k].append(v)
        wandb.log(
            {f'val/{k}': self._convert_video(v)
             for k, v in results.items()},
            step=self.it)
        torch.cuda.empty_cache()
        model.testing = False

    def _log_train(self, out_dict):
        """Log statistics in training to wandb."""
        super()._log_train(out_dict)

        if self.local_rank != 0 or (self.epoch_it + 1) % self.print_iter != 0:
            return

        # log all the lr
        log_dict = {
            'train/lr': get_lr(self.optimizer),
            'train/dec_lr': self.optimizer.param_groups[1]['lr'],
        }
        wandb.log(log_dict, step=self.it)


class VAEMethod(SlotBaseMethod):
    """dVAE and VQVAE model training method."""

    @staticmethod
    def _make_video(video, pred_video):
        """videos are of shape [T, C, H, W]"""
        if isinstance(pred_video, (list, tuple)):
            video_lst = [video, *pred_video]
        else:
            video_lst = [video, pred_video]
        video_lst = [v.cpu() for v in video_lst]
        out = torch.stack(video_lst, dim=1).cpu()  # [T, num_vid, 3, H, W]
        out = torch.stack([
            vutils.make_grid(
                out[i],
                nrow=out.shape[1],
                pad_value=-1.,
            ) for i in range(out.shape[0])
        ])  # [T, 3, H, 2*W]
        return out

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = self.val_loader.dataset
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        results = []
        for idx in sampled_idx:
            data_dict = dst.get_video(idx)
            video = data_dict['video'].float().to(model.device)
            all_recons, bs = [], 100  # a hack to avoid OOM
            for batch_idx in range(0, video.shape[0], bs):
                data_dict = {
                    'img': video[batch_idx:batch_idx + bs],
                    'tau': 1.,
                    'hard': True,
                }
                recon = model(data_dict)['recon']
                all_recons.append(recon)
                torch.cuda.empty_cache()
            recon_video = torch.cat(all_recons, dim=0)
            save_video = self._make_video(video, recon_video)
            results.append(save_video)
        wandb.log({'val/video': self._convert_video(results)}, step=self.it)
        torch.cuda.empty_cache()

    def _training_step_start(self):
        """Things to do at the beginning of every training step."""
        super()._training_step_start()

        if self.params.model != 'dVAE':
            return

        # dVAE: update the tau (gumbel softmax temperature)
        total_steps = self.params.max_epochs * len(self.train_loader)
        decay_steps = self.params.tau_decay_pct * total_steps

        # decay tau
        self.model.module.tau = cosine_anneal(
            self.it,
            start_value=self.params.init_tau,
            final_value=self.params.final_tau,
            start_step=0,
            final_step=decay_steps,
        )

    def _log_train(self, out_dict):
        """Log statistics in training to wandb."""
        super()._log_train(out_dict)

        if self.params.model != 'dVAE':
            return

        if self.local_rank != 0 or (self.epoch_it + 1) % self.print_iter != 0:
            return

        # also log the tau
        wandb.log({'train/gumbel_tau': self.model.module.tau}, step=self.it)
