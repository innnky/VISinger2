import json
import time

import numpy as np
import soundfile
import torch
import tqdm
from scipy.interpolate import interp1d

from utils import utils
from egs.visinger2.models import SynthesizerTrn
from infer import preprocess, cross_fade


trans = -12
speaker = "otto"
ds_path = "infer/share.ds"
config_json = "egs/visinger2/config.json"
checkpoint_path = f"/Volumes/Extend/下载/G_110000.pth"

ds = json.load(open(ds_path))
name = ds_path.split("/")[-1].split(".")[0]
hps = utils.get_hparams_from_file(config_json)
net_g = SynthesizerTrn(hps)
_ = net_g.eval()
_ = utils.load_checkpoint(checkpoint_path, net_g, None)
sample_rate = 44100


result = np.zeros(0)
current_length = 0
for inp in tqdm.tqdm(ds):
    spkid = hps.data.spk2id[speaker]
    f0_seq,pitch, phseq, durations = preprocess(inp)

    pitch = torch.FloatTensor(pitch).unsqueeze(0)

    f0 = torch.FloatTensor(f0_seq).unsqueeze(0)

    text_norm = torch.LongTensor(phseq)
    x_tst = text_norm.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([text_norm.size(0)])
    spk = torch.LongTensor([spkid])
    manual_f0 = torch.FloatTensor(f0).unsqueeze(0)
    manual_dur = torch.LongTensor(durations).unsqueeze(0)
    t1 = time.time()
    infer_res = net_g.infer(x_tst, x_tst_lengths, None,  None,
                         None, gtdur=manual_dur,spk_id=spk,
                         F0=manual_f0 * 2 ** (trans / 12))
    seg_audio = infer_res[0][0, 0].data.float().numpy()
    try:
        offset_ = inp['offset']
    except:
        offset_ = 0
    silent_length = round(offset_ * sample_rate) - current_length
    if silent_length >= 0:
        result = np.append(result, np.zeros(silent_length))
        result = np.append(result, seg_audio)
    else:
        result = cross_fade(result, seg_audio, current_length + silent_length)
    current_length = current_length + silent_length + seg_audio.shape[0]
    print(time.time() - t1)
soundfile.write(f"samples/{speaker}_{name}.wav", result, 44100)
