from videosaur_m.videosaur.modules import timm
from videosaur_m.videosaur.modules.decoders import build as build_decoder
from videosaur_m.videosaur.modules.encoders import build as build_encoder
from videosaur_m.videosaur.modules.groupers import build as build_grouper
from videosaur_m.videosaur.modules.presence_nn import build as build_presence_nn
from videosaur_m.videosaur.modules.initializers import build as build_initializer
from videosaur_m.videosaur.modules.networks import build as build_network
from videosaur_m.videosaur.modules.utils import Resizer, SoftToHardMask
from videosaur_m.videosaur.modules.utils import build as build_utils
from videosaur_m.videosaur.modules.utils import build_module, build_torch_function, build_torch_module
from videosaur_m.videosaur.modules.video import LatentProcessor, MapOverTime, ScanOverTime,IterOverTime,MapOverTime_mask,MapOverTime2,IterOverTime_mask,Map_time_w_modules
from videosaur_m.videosaur.modules.video import Map_time_feature_stack,Map_time_adapt_reshape
from videosaur_m.videosaur.modules.video import build as build_video

__all__ = [
    "build_decoder",
    "build_encoder",
    "build_grouper",
    "build_initializer",
    "build_network",
    "build_utils",
    "build_module",
    "build_torch_module",
    "build_torch_function",
    "timm",
    "MapOverTime",
    "MapOverTime_mask",
    "ScanOverTime",
    "LatentProcessor",
    "Resizer",
    "SoftToHardMask",
]


BUILD_FNS_BY_MODULE_GROUP = {
    "decoders": build_decoder,
    "encoders": build_encoder,
    "groupers": build_grouper,
    "presence_nn": build_presence_nn,
    "initializers": build_initializer,
    "networks": build_network,
    "utils": build_utils,
    "video": build_video,
    "torch": build_torch_function,
    "torch.nn": build_torch_module,
    "nn": build_torch_module,
}
