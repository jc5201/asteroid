import soundfile as sf
import matplotlib.pyplot as plt
import torch
import librosa
import numpy as np
from pathlib import Path

def get_tracks():
    ids = ["id_00", "id_02", "id_04"]
    sources = ["valve"]
    sources_paths = "/dev/shm/mimii/6dB/valve/{id}/normal/*.wav".format(ids)



path = "/dev/shm/mimii/6dB/valve/id_00/normal/00000100.wav"
audio, _ = sf.read("/dev/shm/mimii/6dB/valve/id_00/normal/00000100.wav", always_2d=True)
audio = torch.tensor(audio)

audio = audio[:, :].permute(1,0)
#[channel, time]

rms_fig = librosa.feature.rms(audio)
#[1, 313]
rms_tensor = torch.tensor(rms_fig).reshape(1, -1, 1) 
# [channel, time, 1]
result = rms_tensor.expand(audio.shape[0], -1, 512).reshape(audio.shape[0], -1)[:, :160000]
# [channel, time]

sort_tensor, _ = torch.sort(rms_tensor.reshape(-1))

sort_tensor = sort_tensor[int(rms_tensor.reshape(-1).shape[0]*0.75)]
min_threshold  = torch.mean(sort_tensor)

label = [0 if i<min_threshold else 1 for i in result]

plt.plot(audio)
plt.savefig("audio.png")




