# working_dir_root.py
import os
from working_para.machine_configs import get_machine_path

# Set machine ID (you can keep this variable or switch to purely env vars)
MACHINE_ID = 3 # Or 3
os.environ["MACHINE_ID"] = str(MACHINE_ID)  # Ensure env var is set

# Set default mode if not specified
import_mode = os.environ.get('WORKING_DIR_IMPORT_MODE', 'train_cholec')
 
# Dynamically import based on the mode
if import_mode == 'train_cholec':
    from working_para.working_dir_root_train_cholec_p import *
elif import_mode == 'eval_cholec':
    from working_para.working_dir_root_eval_cholec_full import *
elif import_mode == 'train_miccai':
    from working_para.working_dir_root_train_miccai_p import *
elif import_mode == 'train_thoracic':
    from working_para.working_dir_root_train_Thoracic_p import *
elif import_mode == 'train_movie':
    from working_para.working_dir_root_train_movie import *
elif import_mode == 'train_movid':
    from working_para.working_dir_root_train_movid import *
elif import_mode == 'eval_movie':
    from working_para.working_dir_root_eval_movie import *
elif import_mode == 'eval_movid':
    from working_para.working_dir_root_eval_movid import *
elif import_mode == 'eval_thoracic':
    # from working_para.working_dir_root_eval_Thoracic_p3 import *
    from working_para.working_dir_root_eval_Thoracic_p import *
elif import_mode == 'eval_miccai':
    # from working_para.working_dir_root_eval_miccai_p3 import *
    from working_para.working_dir_root_eval_miccai_p import *
elif import_mode == 'train_ytvos':
    # from working_para.working_dir_root_eval_miccai_p3 import *
    from working_para.working_dir_root_train_ytvos_p import *
elif import_mode == 'eval_ytvos':
    # from working_para.working_dir_root_eval_miccai_p3 import *
    from working_para.working_dir_root_eval_ytvos_p import *
elif import_mode == 'train_ytobj':
    # from working_para.working_dir_root_eval_miccai_p3 import *
    from working_para.working_dir_root_train_ytobj_p import *
elif import_mode == 'eval_ytobj':
    # from working_para.working_dir_root_eval_miccai_p3 import *
    from working_para.working_dir_root_eval_ytobj_p import *
elif import_mode == 'train_pascal':
    # from working_para.working_dir_root_eval_miccai_p3 import *
    from working_para.working_dir_root_train_pascal_p import *
elif import_mode == 'train_poem':
    # from working_para.working_dir_root_eval_miccai_p3 import *
    from working_para.working_dir_root_train_poem_p import *
elif import_mode == 'eval_pascal':
    # from working_para.working_dir_root_eval_miccai_p3 import *
    from working_para.working_dir_root_eval_pascal_p import *
elif import_mode == 'train_coco':
    # from working_para.working_dir_root_eval_miccai_p3 import *
    from working_para.working_dir_root_train_coco_p import *
elif import_mode == 'eval_coco':
    # from working_para.working_dir_root_eval_miccai_p3 import *
    from working_para.working_dir_root_eval_coco_p import *
elif import_mode == 'train_mix':
    # from working_para.working_dir_root_eval_miccai_p3 import *
    from working_para.working_dir_root_train_mix import *
elif import_mode == 'train_mix2':
    # from working_para.working_dir_root_eval_miccai_p3 import *
    from working_para.working_dir_root_train_mix2 import *
elif import_mode == 'train_mix3':
    # from working_para.working_dir_root_eval_miccai_p3 import *
    from working_para.working_dir_root_train_mix3 import *
elif import_mode == 'eval_endovis':
    # from working_para.working_dir_root_eval_miccai_p3 import *
    from working_para.working_dir_root_eval_endovis import *
elif import_mode == 'eval_miccai30fps':
    # from working_para.working_dir_root_eval_miccai_p3 import *
    from working_para.working_dir_root_eval_miccai_30fps import *
elif import_mode == 'train_miccai+cholec':
    # from working_para.working_dir_root_eval_miccai_p3 import *
    from working_para.working_dir_root_train_miccai_cholec_p3 import *
