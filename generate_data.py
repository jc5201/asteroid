import soundfile as sf
import matplotlib.pyplot as plt
import torch
import librosa
import numpy as np
from pathlib import Path


audio, _ = sf.read("/dev/shm/mimii/6dB/slider/id_00/normal/00000011.wav", always_2d=True)
audio = torch.tensor(audio)

audio = audio[:, :].permute(1,0)
#[channel, time]

rms_fig = librosa.feature.rms(audio)
#[1, 313]
rms_tensor = torch.tensor(rms_fig).reshape(1, -1, 1) 
# [channel, time, 1]
result = rms_tensor.expand(-1, -1, 512).reshape(1, -1)[:, :160000]
# [channel, time]


k = int(audio.shape[1]*0.8)
min_threshold, _ = torch.kthvalue(result, k)
print(min_threshold)
label = torch.as_tensor([0.0 if j < min_threshold else 1.0 for j in result[0, :]])

plt.plot(rms_fig.reshape(-1))
plt.savefig("rms.png")

# plt.plot(label)
# plt.savefig("label.png")

# plt.plot(audio[0, :])
# plt.savefig("audio.png")




