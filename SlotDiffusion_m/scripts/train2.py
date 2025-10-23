"""A unified training script for all models used in the SlotDiffusion project."""

import os
import sys
import importlib
import getpass
import platform
import wandb
import torch
from nerv.utils import mkdir_or_exist
from nerv.training import BaseDataModule

# Function to mimic argparse functionality for VSCode debugging
def get_debug_params():
    return {
        'task': 'video_based',  # Task type: 'img_based', 'video_based', 'vp_vqa'
        'params': 'slotdiffusion/video_based/configs/savi_ldm/savi_ldm_movid_params-res128.py',  # Path to parameter file
        'weight': '',  # Optional: path to weights for loading
        'fp16': False,  # Use FP16 precision if needed
        'ddp': False,  # Distributed Data Parallel (DDP) flag
        'cudnn': True,  # Use cudnn benchmark for speed optimization
        'local_rank': 0,  # Rank for distributed training
    }

def main(params, args):
    # Build datamodule
    datasets = build_dataset(params)
    train_set, val_set = datasets[0], datasets[1]
    collate_fn = datasets[2] if len(datasets) == 3 else None
    datamodule = BaseDataModule(
        params,
        train_set=train_set,
        val_set=val_set,
        use_ddp=params.ddp,
        collate_fn=collate_fn,
    )

    # Build model
    model = build_model(params)

    # Create checkpoint directory
    exp_name = os.path.basename(args['params'])
    ckp_path = os.path.join('checkpoint', exp_name, 'models')
    if args['local_rank'] == 0:
        mkdir_or_exist(os.path.dirname(ckp_path))

        # On clusters, quota under user dir is usually limited
        # Save weights in temp space for checkpointing (modify this if you are not running on clusters)
        SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
        if SLURM_JOB_ID and not os.path.exists(ckp_path):
            # Get the current user
            user = getpass.getuser()

            # Temp checkpoint path
            temp_ckp_path = f"/checkpoint/{user}/{SLURM_JOB_ID}/"

            # Handle symbolic links for different platforms
            if platform.system() == "Windows":
                os.system(f'mklink /D {ckp_path} {temp_ckp_path}')
            else:
                os.system(f'ln -s {temp_ckp_path} {ckp_path}')

        # Handle WandB logging
        preemption = False
        if SLURM_JOB_ID and preemption:
            logger_id = logger_name = f'{exp_name}-{SLURM_JOB_ID}'
        else:
            logger_name = exp_name
            logger_id = None
        wandb.init(
            project=params.project,
            name=logger_name,
            id=logger_id,
            dir=ckp_path,
        )

    # Build method
    method = build_method(
        model=model,
        datamodule=datamodule,
        params=params,
        ckp_path=ckp_path,
        local_rank=args['local_rank'],
        use_ddp=args['ddp'],
        use_fp16=args['fp16'],
    )

    # Fit method
    params.san_check_val_step =0
    method.fit(
        resume_from=args['weight'], san_check_val_step=params.san_check_val_step)


if __name__ == "__main__":
    # If debugging in VSCode or directly, you can modify these params.
    is_debugging = 'pydevd' in sys.modules  # Check if running in debug mode (e.g. VSCode)
    Load_para_from_code = True
    if Load_para_from_code or is_debugging:
        print("INFO: Running in debugging mode.")
        args = get_debug_params()  # Set debug params for IDE
    else:
        print("INFO: Running in command line mode.")
        import argparse
        parser = argparse.ArgumentParser(description='SlotDiffusion training')
        parser.add_argument(
            '--task',
            type=str,
            default='img_based',
            choices=['img_based', 'video_based', 'vp_vqa'])
        parser.add_argument('--params', type=str, required=True)
        parser.add_argument('--weight', type=str, default='', help='load weight')
        parser.add_argument('--fp16', action='store_true', help='half-precision')
        parser.add_argument('--ddp', action='store_true', help='DDP training')
        parser.add_argument('--cudnn', action='store_true', help='cudnn benchmark')
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--local-rank', type=int, default=0)
        args = vars(parser.parse_args())  # Convert to dict for ease of use
    print(f"INFO: training model in {args['task']} task!")
    task = importlib.import_module(f"slotdiffusion.{args['task']}")
    build_dataset = task.build_dataset
    build_model = task.build_model
    build_method = task.build_method
    # Load params
    if args['params'].endswith('.py'):
        args['params'] = args['params'][:-3]
    sys.path.append(os.path.dirname(args['params']))
    params_module = importlib.import_module(os.path.basename(args['params']))
    params = params_module.SlotAttentionParams()
    params.ddp = args['ddp']

    if args['fp16']:
        print('INFO: using FP16 training!')
    if args['ddp']:
        print('INFO: using DDP training!')
    if args['cudnn']:
        torch.backends.cudnn.benchmark = True
        print('INFO: using cudnn benchmark!')

    # Start the main training process
    main(params, args)
