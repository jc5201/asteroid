{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "88195427",
   "metadata": {},
   "outputs": [],
   "source": [
    "# import default python-library\n",
    "########################################################################\n",
    "import pickle\n",
    "import os\n",
    "import sys\n",
    "import glob\n",
    "########################################################################\n",
    "\n",
    "\n",
    "########################################################################\n",
    "# import additional python-library\n",
    "########################################################################\n",
    "import numpy\n",
    "import librosa\n",
    "import librosa.core\n",
    "import librosa.feature\n",
    "import yaml\n",
    "import logging\n",
    "# from import\n",
    "from tqdm import tqdm\n",
    "from sklearn import metrics\n",
    "from keras.models import Model\n",
    "from keras.layers import Input, Dense\n",
    "\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "from asteroid.models import XUMXControl\n",
    "import museval"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "bf1ffed8",
   "metadata": {},
   "outputs": [],
   "source": [
    "#set machine and id\n",
    "MACHINE = 'valve'\n",
    "S1 = 'id_04'\n",
    "S2 = 'id_06'\n",
    "FILE = 'test2.pth'\n",
    "model_path = '/hdd/hdd1/lyj/xumx/output_w_cont_valve3/checkpoints/epoch=998-step=44954.ckpt'\n",
    "\n",
    "\n",
    "# choose wave file and status\n",
    "wav_num1 = \"00000001\"\n",
    "wav_num2 = \"00000010\"\n",
    "status_s1 = \"abnormal\"\n",
    "status_s2 = \"normal\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "454f2484",
   "metadata": {},
   "outputs": [],
   "source": [
    "# load weight of AE a\n",
    "ae1_path = '/hdd/hdd1/lyj/xumx/ae/cont/{machine}/{source}/{file}'.format(machine = MACHINE, source = S1, file = FILE)\n",
    "ae2_path = '/hdd/hdd1/lyj/xumx/ae/cont/{machine}/{source}/{file}'.format(machine = MACHINE, source = S2, file = FILE)\n",
    "\n",
    "# set source path \n",
    "source1_path = \"/dev/shm/mimii/6dB/{machine}/{source}/{status}/{num}.wav\".format(machine = MACHINE, source = S1, status= status_s1, num = wav_num1)\n",
    "source2_path = \"/dev/shm/mimii/6dB/{machine}/{source}/{status}/{num}.wav\".format(machine = MACHINE, source = S2, status = status_s2, num = wav_num2)\n",
    "\n",
    "machine_types = [S1, S2]\n",
    "source_path = [source1_path, source2_path]\n",
    "\n",
    "threshold_lst = []\n",
    "num_eval_normal = 250"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "62b184d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# pickle I/O\n",
    "def save_pickle(filename, save_data):\n",
    "    with open(filename, 'wb') as sf:\n",
    "        pickle.dump(save_data, sf)\n",
    "\n",
    "\n",
    "def load_pickle(filename):\n",
    "    with open(filename, 'rb') as lf:\n",
    "        load_data = pickle.load(lf)\n",
    "    return load_data\n",
    "\n",
    "\n",
    "# wav file Input\n",
    "def file_load(wav_name, mono=False):\n",
    "    return librosa.load(wav_name, sr=None, mono=mono)\n",
    "        \n",
    "\n",
    "\n",
    "def demux_wav(wav_name, channel=1):\n",
    "   \n",
    "    multi_channel_data, sr = file_load(wav_name)\n",
    "    if multi_channel_data.ndim <= 1:\n",
    "        return sr, multi_channel_data\n",
    "\n",
    "    return sr, numpy.array(multi_channel_data)[:channel, :]\n",
    "\n",
    "\n",
    "########################################################################\n",
    "# feature extractor\n",
    "########################################################################\n",
    "\n",
    "def file_to_wav(file_name):\n",
    "    sr, y = demux_wav(file_name, channel=2)\n",
    "    return sr, y\n",
    "\n",
    "def wav_to_vector_array(sr, y,\n",
    "                         n_mels=64,\n",
    "                         frames=5,\n",
    "                         n_fft=1024,\n",
    "                         hop_length=512,\n",
    "                         power=2.0):\n",
    "    \"\"\"\n",
    "    convert file_name to a vector array.\n",
    "    file_name : str\n",
    "        target .wav file\n",
    "    return : numpy.array( numpy.array( float ) )\n",
    "        vector array\n",
    "        * dataset.shape = (dataset_size, fearture_vector_length)\n",
    "    \"\"\"\n",
    "    # 01 calculate the number of dimensions\n",
    "    dims = n_mels * frames\n",
    "\n",
    "    # 02 generate melspectrogram using librosa (**kwargs == param[\"librosa\"])\n",
    "    mel_spectrogram = librosa.feature.melspectrogram(y=y,\n",
    "                                                     sr=sr,\n",
    "                                                     n_fft=n_fft,\n",
    "                                                     hop_length=hop_length,\n",
    "                                                     n_mels=n_mels,\n",
    "                                                     power=power)\n",
    "\n",
    "    # 03 convert melspectrogram to log mel energy\n",
    "    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)\n",
    "\n",
    "    # 04 calculate total vector size\n",
    "    vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1\n",
    "\n",
    "    # 05 skip too short clips\n",
    "    if vectorarray_size < 1:\n",
    "        return numpy.empty((0, dims), float)\n",
    "\n",
    "    # 06 generate feature vectors by concatenating multi_frames\n",
    "    vectorarray = numpy.zeros((vectorarray_size, dims), float)\n",
    "    for t in range(frames):\n",
    "        vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T\n",
    "\n",
    "    return vectorarray\n",
    "\n",
    "\n",
    "def bandwidth_to_max_bin(rate, n_fft, bandwidth):\n",
    "    freqs = numpy.linspace(0, float(rate) / 2, n_fft // 2 + 1, endpoint=True)\n",
    "\n",
    "    return numpy.max(numpy.where(freqs <= bandwidth)[0]) + 1\n",
    "\n",
    "\n",
    "class XUMXSystem(torch.nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.model = None\n",
    "\n",
    "\n",
    "def xumx_model(path):\n",
    "    \n",
    "    x_unmix = XUMXControl(\n",
    "        window_length=4096,\n",
    "        input_mean=None,\n",
    "        input_scale=None,\n",
    "        nb_channels=2,\n",
    "        hidden_size=512,\n",
    "        in_chan=4096,\n",
    "        n_hop=1024,\n",
    "        sources=['id_00', 'id_02'],\n",
    "        max_bin=bandwidth_to_max_bin(16000, 4096, 16000),\n",
    "        bidirectional=True,\n",
    "        sample_rate=16000,\n",
    "        spec_power=1,\n",
    "        return_time_signals=True,\n",
    "    )\n",
    "\n",
    "    conf = torch.load(path, map_location=\"cpu\")\n",
    "\n",
    "    system = XUMXSystem()\n",
    "    system.model = x_unmix\n",
    "\n",
    "    system.load_state_dict(conf['state_dict'], strict=False)\n",
    "\n",
    "    return system.model\n",
    "\n",
    "\n",
    "machine_types = [S1, S2]\n",
    "num_eval_normal = 250\n",
    "\n",
    "\n",
    "def train_list_to_vector_array(file_list,\n",
    "                         msg=\"calc...\",\n",
    "                         n_mels=64,\n",
    "                         frames=5,\n",
    "                         n_fft=1024,\n",
    "                         hop_length=512,\n",
    "                         power=2.0,\n",
    "                         target_source=None):\n",
    "    \n",
    "    # 01 calculate the number of dimensions\n",
    "    dims = n_mels * frames\n",
    "\n",
    "    # 02 loop of file_to_vectorarray\n",
    "    for idx in tqdm(range(len(file_list)), desc=msg):\n",
    "        active_label_sources = {}\n",
    "        mixture_y = 0\n",
    "        target_type = os.path.split(os.path.split(os.path.split(file_list[idx])[0])[0])[1]\n",
    "        if target_source is not None:\n",
    "            target_idx = machine_types.index(target_source)\n",
    "        else:\n",
    "            target_idx = machine_types.index(target_type)\n",
    "        for machine in machine_types:\n",
    "            filename = file_list[idx].replace(target_type, machine)\n",
    "            sr, y = file_to_wav(filename)\n",
    "            ##############################################################\n",
    "            #generate control signal \n",
    "            label = generate_label(y)\n",
    "            active_label_sources[machine] = label\n",
    "            ##############################################################\n",
    "            mixture_y = mixture_y + y\n",
    "            \n",
    "        active_labels = torch.stack([active_label_sources[src] for src in machine_types])\n",
    "        _, time = sep_model(torch.Tensor(mixture_y).unsqueeze(0).cuda(), active_labels.unsqueeze(0).cuda())\n",
    "        \n",
    "        # [src, b, ch, time]\n",
    "        ys = time[target_idx, 0, 0, :].detach().cpu().numpy()\n",
    "        \n",
    "        vector_array = wav_to_vector_array(sr, ys,\n",
    "                                            n_mels=n_mels,\n",
    "                                            frames=frames,\n",
    "                                            n_fft=n_fft,\n",
    "                                            hop_length=hop_length,\n",
    "                                            power=power)\n",
    "\n",
    "        if idx == 0:\n",
    "            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)\n",
    "\n",
    "        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array\n",
    "\n",
    "    return dataset\n",
    "\n",
    "\n",
    "class AEDataset(torch.utils.data.Dataset):\n",
    "    def __init__(self, \n",
    "            sep_model, \n",
    "            file_list,\n",
    "            param,\n",
    "            target_source=None,\n",
    "            ):\n",
    "        self.sep_model = sep_model\n",
    "        self.file_list = file_list\n",
    "        self.target_source = target_source\n",
    "\n",
    "        self.data_vector = train_list_to_vector_array(self.file_list,\n",
    "                                            msg=\"generate train_dataset\",\n",
    "                                            n_mels=param[\"feature\"][\"n_mels\"],\n",
    "                                            frames=param[\"feature\"][\"frames\"],\n",
    "                                            n_fft=param[\"feature\"][\"n_fft\"],\n",
    "                                            hop_length=param[\"feature\"][\"hop_length\"],\n",
    "                                            power=param[\"feature\"][\"power\"],\n",
    "                                            target_source=target_source)\n",
    "        \n",
    "    \n",
    "    def __getitem__(self, index):\n",
    "        return torch.Tensor(self.data_vector[index, :])\n",
    "    \n",
    "    def __len__(self):\n",
    "        return self.data_vector.shape[0]\n",
    "\n",
    "\n",
    "def dataset_generator(target_dir,\n",
    "                      normal_dir_name=\"normal\",\n",
    "                      abnormal_dir_name=\"abnormal\",\n",
    "                      ext=\"wav\"):\n",
    "\n",
    "    # 01 normal list generate\n",
    "    normal_files = sorted(glob.glob(\n",
    "        os.path.abspath(\"{dir}/{normal_dir_name}/*.{ext}\".format(dir=target_dir,\n",
    "                                                                 normal_dir_name=normal_dir_name,\n",
    "                                                                ext=ext))))\n",
    "    normal_len = [len(glob.glob(\n",
    "        os.path.abspath(\"{dir}/{normal_dir_name}/*.{ext}\".format(dir=target_dir.replace(S1, mt),\n",
    "                                                                 normal_dir_name=normal_dir_name,\n",
    "                                                                 ext=ext)))) for mt in machine_types]   #dataset 중에서 가장 짧은 것 기준\n",
    "    normal_len = min(min(normal_len), len(normal_files))\n",
    "    normal_files = normal_files[:normal_len]\n",
    "\n",
    "\n",
    "    normal_labels = numpy.zeros(len(normal_files))\n",
    "\n",
    "    # 02 abnormal list generate\n",
    "    abnormal_files = sorted(glob.glob(\n",
    "        os.path.abspath(\"{dir}/{abnormal_dir_name}/*.{ext}\".format(dir=target_dir,\n",
    "                                                                   abnormal_dir_name=abnormal_dir_name,\n",
    "                                                                   ext=ext))))\n",
    "    abnormal_files.extend(sorted(glob.glob(\n",
    "        os.path.abspath(\"{dir}/{abnormal_dir_name}/*.{ext}\".format(dir=target_dir.replace(S1, S2),\n",
    "                                                                   abnormal_dir_name=abnormal_dir_name,\n",
    "                                                                 ext=ext)))))                                               \n",
    "    abnormal_labels = numpy.ones(len(abnormal_files))\n",
    "\n",
    "    # 03 separate train & eval\n",
    "    train_files = normal_files[num_eval_normal:]\n",
    "    train_labels = normal_labels[num_eval_normal:]\n",
    "    eval_normal_files = sum([[fan_file.replace(S1, machine_type) for fan_file in normal_files[:num_eval_normal]] for machine_type in machine_types], [])\n",
    "    eval_files = numpy.concatenate((eval_normal_files, abnormal_files), axis=0)\n",
    "    eval_labels = numpy.concatenate((normal_labels[:num_eval_normal], normal_labels[:num_eval_normal], abnormal_labels), axis=0)  ##TODO \n",
    "\n",
    "    return train_files, train_labels, eval_files, eval_labels\n",
    "\n",
    "\n",
    "########################################################################\n",
    "# keras model\n",
    "########################################################################\n",
    "def keras_model(inputDim):\n",
    "    \"\"\"\n",
    "    define the keras model\n",
    "    the model based on the simple dense auto encoder (64*64*8*64*64)\n",
    "    \"\"\"\n",
    "    inputLayer = Input(shape=(inputDim,))\n",
    "    h = Dense(64, activation=\"relu\")(inputLayer)\n",
    "    h = Dense(64, activation=\"relu\")(h)\n",
    "    h = Dense(8, activation=\"relu\")(h)\n",
    "    h = Dense(64, activation=\"relu\")(h)\n",
    "    h = Dense(64, activation=\"relu\")(h)\n",
    "    h = Dense(inputDim, activation=None)(h)\n",
    "\n",
    "    return Model(inputs=inputLayer, outputs=h)\n",
    "\n",
    "class TorchModel(nn.Module):\n",
    "    def __init__(self, dim_input):\n",
    "        super(TorchModel,self).__init__()\n",
    "        self.ff = nn.Sequential(\n",
    "            nn.Linear(dim_input, 64),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(64, 64),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(64, 8),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(8, 64),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(64, 64),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(64, dim_input),\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.ff(x)\n",
    "        return x\n",
    "\n",
    "\n",
    "def generate_label(y):\n",
    "    rms_fig = librosa.feature.rms(y)\n",
    "    rms_tensor = torch.tensor(rms_fig).reshape(1, -1, 1)\n",
    "    rms_trim = rms_tensor.expand(-1, -1, 512).reshape(1, -1)[:, :160000]\n",
    "\n",
    "    if MACHINE == 'valve':\n",
    "        k = int(y.shape[1]*0.8)\n",
    "        min_threshold, _ = torch.kthvalue(rms_trim, k)\n",
    "    else:\n",
    "        min_threshold = (torch.max(rms_trim) + torch.min(rms_trim))/2\n",
    "    label = (rms_trim > min_threshold).type(torch.float)\n",
    "    label = label.expand(y.shape[0], -1)\n",
    "    return label\n",
    "########################################################################\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "db24328c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load separation model\n",
    "sep_model = xumx_model(model_path)\n",
    "sep_model.eval()\n",
    "sep_model = sep_model.cuda()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "7d902f30",
   "metadata": {},
   "outputs": [],
   "source": [
    "with open(\"baseline.yaml\") as stream:\n",
    "    param = yaml.safe_load(stream)\n",
    "train_files, train_labels, eval_files, eval_labels = dataset_generator(\"/dev/shm/mimii/6dB/{machine}/{source}\".format(machine = MACHINE, source = S1))\n",
    "dim_input = 320"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "04e7f59e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# load auto encoder\n",
    "ae_model1, ae_model2 = TorchModel(dim_input), TorchModel(dim_input)\n",
    "checkpoint_s1, checkpoint_s2 = torch.load(ae1_path), torch.load(ae2_path)\n",
    "ae_model1.load_state_dict(checkpoint_s1)\n",
    "ae_model2.load_state_dict(checkpoint_s2)\n",
    "ae_model1.eval()\n",
    "ae_model2.eval()\n",
    "ae_model1, ae_model2 = ae_model1.cuda(), ae_model2.cuda()\n",
    "anomaly_model = {S1:ae_model1, S2:ae_model2}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "de208e87",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  0%|                                                                                                                                                                                                             | 0/740 [00:00<?, ?it/s]/home/lyj/.conda/envs/xumx/lib/python3.8/site-packages/torch/functional.py:770: UserWarning: istft will require a complex-valued input tensor in a future PyTorch release. Matching the output from stft with return_complex=True.  (Triggered internally at  /opt/conda/conda-bld/pytorch_1646755903507/work/aten/src/ATen/native/SpectralOps.cpp:950.)\n",
      "  return _VF.istft(input, n_fft, hop_length, win_length, window, center,  # type: ignore[attr-defined]\n",
      "100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 740/740 [00:32<00:00, 22.93it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SCORE: 0.36373333333333335\n",
      "SCORE: 0.5076666666666667\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "# find best threshold\n",
    "y_pred = numpy.array([0. for k in eval_labels])\n",
    "y_true = numpy.array(eval_labels)\n",
    "\n",
    "eval_types = {mt: [] for mt in machine_types}\n",
    "\n",
    "for num, file_name in tqdm(enumerate(eval_files), total=len(eval_files)):\n",
    "    machine_type = os.path.split(os.path.split(os.path.split(file_name)[0])[0])[1]\n",
    "    target_idx = machine_types.index(machine_type)  \n",
    "    y_raw = {}\n",
    "    mixture_y = 0\n",
    "    active_label_sources = {}\n",
    "    for normal_type in machine_types:\n",
    "        if normal_type == machine_type:\n",
    "            continue\n",
    "        normal_file_name = file_name.replace(machine_type, normal_type).replace('abnormal', 'normal')\n",
    "        sr, y = file_to_wav(normal_file_name)\n",
    "        label = generate_label(y)\n",
    "        active_label_sources[normal_type] = label\n",
    "        mixture_y += y \n",
    "        y_raw[normal_type] = y\n",
    "\n",
    "    sr, y = file_to_wav(file_name)\n",
    "    label = generate_label(y)\n",
    "    active_label_sources[machine_type] = label \n",
    "    mixture_y += y\n",
    "    y_raw[machine_type] = y\n",
    "\n",
    "    active_labels = torch.stack([active_label_sources[src] for src in machine_types])\n",
    "    _, time = sep_model(torch.Tensor(mixture_y).unsqueeze(0).cuda(), active_labels.unsqueeze(0).cuda())\n",
    "    # [src, b, ch, time]\n",
    "    ys = time[target_idx, 0, 0, :].detach().cpu().numpy()\n",
    "\n",
    "    data = wav_to_vector_array(sr, ys,\n",
    "                                n_mels=param[\"feature\"][\"n_mels\"],\n",
    "                                frames=param[\"feature\"][\"frames\"],\n",
    "                                n_fft=param[\"feature\"][\"n_fft\"],\n",
    "                                hop_length=param[\"feature\"][\"hop_length\"],\n",
    "                                power=param[\"feature\"][\"power\"])\n",
    "    data = torch.Tensor(data).cuda()\n",
    "    error = torch.mean(((data - anomaly_model[machine_type](data)) ** 2), dim=1)\n",
    "\n",
    "    y_pred[num] = torch.mean(error).detach().cpu().numpy()\n",
    "    eval_types[machine_type].append(num)\n",
    "\n",
    "for machine_type in machine_types:\n",
    "    fpr, tpr,threshold = metrics.roc_curve(y_true[eval_types[machine_type]], y_pred[eval_types[machine_type]])\n",
    "    score = metrics.roc_auc_score(y_true[eval_types[machine_type]], y_pred[eval_types[machine_type]])\n",
    "    print(\"SCORE:\", score)\n",
    "    j = tpr-fpr\n",
    "    idx = numpy.argmax(j)\n",
    "    threshold_lst.append(numpy.array(threshold[idx]).astype('float'))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "8c066d63",
   "metadata": {},
   "outputs": [],
   "source": [
    "def source_separation():\n",
    "    active_label_sources = {}\n",
    "    separated_dataset = {}\n",
    "    mixture_y = 0\n",
    "\n",
    "    for idx, machine in enumerate(machine_types):\n",
    "        file_name = source_path[idx]\n",
    "        sr, y = file_to_wav(file_name)\n",
    "\n",
    "        label = generate_label(y)\n",
    "        active_label_sources[machine] = label\n",
    "\n",
    "        mixture_y = mixture_y + y\n",
    "\n",
    "    active_labels = torch.stack([active_label_sources[src] for src in machine_types])   \n",
    "    _, time = sep_model(torch.Tensor(mixture_y).unsqueeze(0).cuda(), active_labels.unsqueeze(0).cuda())\n",
    "\n",
    "    for idx, machine in enumerate(machine_types):\n",
    "        ys = time[idx, 0, 0, :].detach().cpu().numpy()\n",
    "        \n",
    "        \n",
    "        vector_array = wav_to_vector_array(sr, ys,\n",
    "                                                n_mels=64,\n",
    "                                                frames=5,\n",
    "                                                n_fft=1024,\n",
    "                                                hop_length=512,\n",
    "                                                power=2.0)\n",
    "        seprated_data = torch.Tensor(vector_array).cuda()\n",
    "        separated_dataset[machine] = seprated_data\n",
    "    return separated_dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "ee6d2be6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Find anomaly source\n",
    "def anomaly_detection(anomaly_model, threshold_lst):\n",
    "    result_lst = []\n",
    "    separated_dataset = source_separation()\n",
    "    for idx, machine in enumerate(machine_types):\n",
    "        error = torch.mean(torch.mean((separated_dataset[machine] - anomaly_model[machine](separated_dataset[machine])) ** 2 , dim = 1))\n",
    "        threshold = torch.from_numpy(threshold_lst[idx]).cuda()\n",
    "        if error >= threshold:\n",
    "            result = \"abnormal\"\n",
    "        else:\n",
    "            result = \"normal\"\n",
    "        result_lst.append(result)                      \n",
    "    return result_lst"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "7c7b4eaf",
   "metadata": {},
   "outputs": [],
   "source": [
    "anomaly_result = anomaly_detection(anomaly_model, threshold_lst)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "18acaff1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Anomaly classification result:\n",
      "valve_id_04 is abnormal, and is detected to be normal\n",
      "valve_id_06 is normal, and is detected to be normal\n"
     ]
    }
   ],
   "source": [
    "print(\"Anomaly classification result:\")\n",
    "print(\"{machine}_{source} is {status}, and is detected to be {status_pred}\".format(machine = MACHINE, source = S1, status = status_s1, status_pred = anomaly_result[0]))\n",
    "print(\"{machine}_{source} is {status}, and is detected to be {status_pred}\".format(machine = MACHINE, source = S2, status = status_s2, status_pred = anomaly_result[1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4c22354c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
