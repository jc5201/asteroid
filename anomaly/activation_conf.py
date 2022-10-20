import scipy.signal
import numpy as np
import librosa
import os
import torch.nn as nn
import torch
import librosa

def compute_activation_confidence(audio, rate, win_len=2048, lpf_cutoff=0.075,
                                 theta=0.15, var_lambda=20.0,
                                 amplitude_threshold=0.01):
    """Create the activation confidence annotation for a multitrack. The final
    activation matrix is computed as:
        `C[i, t] = 1 - (1 / (1 + e**(var_lambda * (H[i, t] - theta))))`
    where H[i, t] is the energy of stem `i` at time `t`
    Parameters
    ----------
    mtrack : MultiTrack
        Multitrack object
    win_len : int, default=4096
        Number of samples in each window
    lpf_cutoff : float, default=0.075
        Lowpass frequency cutoff fraction
    theta : float
        Controls the threshold of activation.
    var_labmda : float
        Controls the slope of the threshold function.
    amplitude_threshold : float
        Energies below this value are set to 0.0
    Returns
    -------
    C : np.array
        Array of activation confidence values shape (n_conf, n_stems)
    stem_index_list : list
        List of stem indices in the order they appear in C
    """
    H = []
    frames = 5
    # MATLAB equivalent to @hanning(win_len)
    
    audio = audio[0, :]
    #audio = librosa.util.normalize(audio)
    win = scipy.signal.windows.hann(win_len + 2)[1:-1]
   
    H.append(track_energy(audio.T, win_len, win))

    # list to numpy array
    H = np.array(H)
    # normalization (to overall energy and # of sources)
    E0 = np.sum(H, axis=0)
   
    H =  H / np.max(E0)
    # binary thresholding for low overall energy events
    H[:, E0 < amplitude_threshold] = 0.0

    # LP filter
    b, a = scipy.signal.butter(2, lpf_cutoff, 'low')
    H = scipy.signal.filtfilt(b, a, H, axis=1)

    # logistic function to semi-binarize the output; confidence value
    C = 1.0 - (1.0 / (1.0 + np.exp(np.dot(var_lambda, (H - theta)))))
    C = torch.Tensor(C)
        
    # generate spec_label
    spec_label_tmp = nn.functional.adaptive_avg_pool1d(C, 313).repeat(2, 1)
    spec_label = torch.stack([torch.tensor(spec_label_tmp[:, i:i+spec_label_tmp.shape[1]- frames + 1]) for i in range(frames)], dim = 2)

    return C, spec_label


def track_energy(wave, win_len, win):
    """Compute the energy of an audio signal
    Parameters
    ----------
    wave : np.array
        The signal from which to compute energy
    win_len: int
        The number of samples to use in energy computation
    win : np.array
        The windowing function to use in energy computation
    Returns
    -------
    energy : np.array
        Array of track energy
    """
    hop_len = win_len // 2

    wave = np.lib.pad(
        wave, pad_width=(win_len-hop_len, 0), mode='constant', constant_values=0
    )

    # post padding
    wave = librosa.util.fix_length(
        wave, int(win_len * np.ceil(len(wave) / win_len))
    )
    # cut into frames
    wavmat = librosa.util.frame(wave, frame_length=win_len, hop_length=hop_len)

    # Envelope follower
    wavmat = hwr(wavmat) ** 0.5  # half-wave rectification + compression

    return np.mean((wavmat.T * win), axis=1)


def hwr(x):
    """ Half-wave rectification.
    Parameters
    ----------
    x : array-like
        Array to half-wave rectify
    Returns
    -------
    x_hwr : array-like
        Half-wave rectified array
    """
    return (x + np.abs(x)) / 2

