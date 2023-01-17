import json
import os
import time
import re

import numpy as np
import soundfile
import torch
import tqdm
from scipy.interpolate import interp1d

from utils import utils
from egs.visinger2.models import SynthesizerTrn
from infer import preprocess, cross_fade, infer_ds

trans = -12
speaker = "otto"
ds_path = "infer/share.ds"
config_json = "egs/visinger2/config.json"
checkpoint_path = f"/Volumes/Extend/下载/G_157000.pth"
file_name = os.path.splitext(os.path.basename(ds_path))[0]
step = re.findall(r'G_(\d+)\.pth', checkpoint_path)[0]


ds = json.load(open(ds_path))
hps = utils.get_hparams_from_file(config_json)
net_g = SynthesizerTrn(hps)
_ = net_g.eval()
_ = utils.load_checkpoint(checkpoint_path, net_g, None)

audio = infer_ds(net_g, hps, ds, speaker, trans)
soundfile.write(f"samples/{speaker}_{file_name}_{step}step.wav", audio, 44100)
