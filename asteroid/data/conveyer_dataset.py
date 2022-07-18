from pathlib import Path
import torch.utils.data
import random
import torch
import tqdm
import soundfile as sf

from torchaudio import transforms
import librosa
from itertools import product
import numpy as np

class ConveyerDataset(torch.utils.data.Dataset):
   
    dataset_name = "Conveyer"

    def __init__(
        self,
        root,
        sources=["close", "far"],
        targets=None,
        suffix=".wav",
        split="0dB",
        subset=None,
        segment=None,
        samples_per_track=2,
        random_segments=False,
        random_track_mix=False,
        source_augmentations=lambda audio: audio,
        sample_rate=16000,
        normal=True,
        use_control=False,
        task_random = False,
        source_random = False
    ):

        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.segment = segment
        self.random_track_mix = random_track_mix
        self.random_segments = random_segments
        self.source_augmentations = source_augmentations
        self.sources = sources
        self.targets = targets
        self.suffix = suffix
        self.subset = subset
        self.samples_per_track = samples_per_track
        self.normal = normal
        self.tracks = list(self.get_tracks())
        #print(self.tracks)
        if not self.tracks:
            raise RuntimeError("No tracks found.")
        self.use_control = use_control 
        self.normal = True
        self.task_random = task_random
        self.source_random = source_random

    def __getitem__(self, index):
       
        audio_sources = {}
        active_label_sources = {}

        if self.source_random:
            sources_tmp = random.sample(self.sources, 2)
            target_tmp = sources_tmp
        else:
            sources_tmp = self.sources
            target_tmp = self.targets

        # get track_id
        track_id = index // self.samples_per_track
        if self.random_segments:
            start = random.uniform(0, self.tracks[track_id]["min_duration"] - self.segment)
        else:
            start = 0

        # load sources
        for i, source in enumerate(sources_tmp):
            # optionally select a random track for each source
            if self.random_track_mix:
                # load a different track
                track_id = random.choice(range(len(self.tracks)))
                if self.random_segments:
                    start = random.uniform(0, self.tracks[track_id]["min_duration"] - self.segment)

            # loads the full track duration
            start_sample = int(start * self.sample_rate)
            # check if dur is none
            if self.segment:
                # stop in soundfile is calc in samples, not seconds
                stop_sample = start_sample + int(self.segment * self.sample_rate)
            else:
                # set to None for reading complete file
                stop_sample = None

            # load actual audio
            np_audio, _ = sf.read(
                Path(self.tracks[track_id]["source_paths"][i]),
                always_2d=True,
                start=start_sample,
                stop=stop_sample,
            )
            # convert to torch tensor
            audio = torch.tensor(np_audio.T, dtype=torch.float)[:, :]
            # apply source-wise augmentations
            audio = self.source_augmentations(audio)

            #apply mask
            audio_len = audio.shape[1]
            mask_len = random.randrange(int(audio_len * 0.8))
            if i == 0:
                start_point = 0
            else:
                start_point = audio_len - mask_len
            torch.clamp_(audio[:, start_point:start_point + mask_len], min=-0.01, max=0.01)
            audio_sources[source] = audio  
            #[channel, time]
            
            if self.use_control:
                rms_fig = librosa.feature.rms(np.transpose(np_audio)) #[1, 313]
                rms_tensor = torch.tensor(rms_fig).reshape(1, -1, 1)
                # [channel, time, 1]
                rms_trim = rms_tensor.expand(-1, -1, 512).reshape(1, -1)[:, :160000]
                # [channel, time]

                if self.normal:
                    k = int(audio.shape[1]*0.8)
                    min_threshold, _ = torch.kthvalue(rms_trim, k)

                    label = (rms_trim > min_threshold).type(torch.float) 
                    # label = torch.as_tensor([0.0 if j < min_threshold else 1.0 for j in rms_trim[0, :]])
                    label = label.expand(audio.shape[0], -1)
                    active_label_sources[source] = label
                    #[channel, time]

                else:
                    label = random.choices([0,1], k=10)
                    label = label.expand(audio.shape[1])
                    label = label.expand(audio.shape[0], -1)
                    active_label_sources[source] = label
            

        # apply linear mix over source index=0
        # make mixture for i-th channel and use 0-th chnnel as gt
        if self.task_random:
            targets = target_tmp.copy()
            random.shuffle(targets)       
        else:
            targets = target_tmp
        audioes = torch.stack([audio_sources[src] for src in targets])
        audio_mix = torch.stack([audioes[i, 0:2, :] for i in range(len(sources_tmp))]).sum(0)

        #use different channel for two different valves
        if targets:
            audio_sources = audioes[:, 0:2, :]
            #audio_sources[1, :, :] = audioes[1, 2:4, :]

        if self.use_control:
            active_labels = torch.stack([active_label_sources[src] for src in targets])
            # [source, channel, time]
            if targets:
                active_labels = active_labels[:, 0:2, :]
            return audio_mix, audio_sources, active_labels

        return audio_mix, audio_sources

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def get_tracks(self):
        """Loads input and output tracks"""
        p = Path(self.root)
        pp = []
        pp.extend(p.glob(f'close/{"normal" if self.normal else "abnormal"}/*.WAV'))
        print("=============================================")
        for track_path in tqdm.tqdm(pp):
            if self.subset and track_path.stem not in self.subset:
                # skip this track
                continue
           
            source_paths = [Path(str(track_path).replace(self.sources[0], s)) for s in self.sources]
            print(source_paths)
            if not all(sp.exists() for sp in source_paths):
                print("Exclude track due to non-existing source", track_path)
                continue

            # get metadata
            infos = list(map(sf.info, source_paths))
            if not all(i.samplerate == self.sample_rate for i in infos):
                print("Exclude track due to different sample rate ", track_path)
                continue
          
            if self.segment is not None:
                # get minimum duration of track
                min_duration = min(i.duration for i in infos)
                if min_duration > self.segment:
                    yield ({"path": track_path, "min_duration": min_duration, "source_paths": source_paths})
            else:
                yield ({"path": track_path, "min_duration": None, "source_paths": source_paths})
