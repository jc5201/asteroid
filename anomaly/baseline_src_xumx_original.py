#!/usr/bin/env python

########################################################################
# import default python-library
########################################################################
import pickle
import os
import sys
import glob

import numpy
import numpy as np
import librosa
import librosa.core
import librosa.feature
import yaml
import logging
import random

from tqdm import tqdm
from sklearn import metrics


import torch
import torch.nn as nn
from asteroid.models import XUMXControl
import museval


from utils import *
from model import TorchModel
########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.3"
########################################################################


########################################################################
# choose machine type and id
S1 = 'id_00'
S2 = 'id_02'
MACHINE = 'slider'
FILE = 'slider_id00_id02_original.pth'
xumx_model_path = '/hdd/hdd1/lyj/xumx/output_w_cont_slider2/checkpoints/epoch=998-step=66932.ckpt'
ae_path_base = '/hdd/hdd1/lyj/xumx/ae/cont'

machine_types = [S1, S2]
num_eval_normal = 250

########################################################################


########################################################################
# feature extractor
########################################################################


def generate_label(y):
        # np, [c, t]
        channels = y.shape[0]
    rms_fig = librosa.feature.rms(y=y)
        #[c, 1, 313]

        rms_tensor = torch.tensor(rms_fig).permute(0, 2, 1)
        # [channel, time, 1]
        rms_trim = rms_tensor.expand(-1, -1, 512).reshape(channels, -1)[:, :160000]
        # [channel, time]

    if MACHINE == 'valve':
        k = int(y.shape[1]*0.8)
            min_threshold, _ = torch.kthvalue(rms_trim[0, :], k)
    else:
        min_threshold = (torch.max(rms_trim) + torch.min(rms_trim))/2
    
    label = (rms_trim > min_threshold).type(torch.float)
        #[channel, time]
    return label

def train_file_to_mixture_wav_label(filename):
    machine_type = os.path.split(os.path.split(os.path.split(filename)[0])[0])[1]
    ys = 0
    active_label_sources = {}
    for machine in machine_types:
        src_filename = filename.replace(machine_type, machine)
        sr, y = file_to_wav_stereo(src_filename)
        active_label_sources[machine] = generate_label(y)
        ys = ys + y

    return sr, ys, active_label_sources


def eval_file_to_mixture_wav_label(filename):
    machine_type = os.path.split(os.path.split(os.path.split(filename)[0])[0])[1]
    ys = 0
    gt_wav = {}
    active_label_sources = {}
    for normal_type in machine_types:
        if normal_type == machine_type:
            src_filename = filename
        else:
            src_filename = filename.replace(machine_type, normal_type).replace('abnormal', 'normal')
        sr, y = file_to_wav_stereo(src_filename)
        ys = ys + y
        active_label_sources[normal_type] = generate_label(y)
        gt_wav[normal_type] = y
    
    return sr, ys, gt_wav, active_label_sources

def get_overlap_ratio(signal1, signal2):
    return torch.sum(torch.logical_and(signal1, signal2)) / torch.sum(torch.logical_or(signal1, signal2))


class XUMXSystem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None


def xumx_model(path):
    
    x_unmix = XUMXControl(
        window_length=4096,
        input_mean=None,
        input_scale=None,
        nb_channels=2,
        hidden_size=512,
        in_chan=4096,
        n_hop=1024,
        sources=['s1', 's2'],
        max_bin=bandwidth_to_max_bin(16000, 4096, 16000),
        bidirectional=True,
        sample_rate=16000,
        spec_power=1,
        return_time_signals=True,
    )

    conf = torch.load(path, map_location="cpu")

    system = XUMXSystem()
    system.model = x_unmix

    system.load_state_dict(conf['state_dict'], strict=False)

    return system.model



def train_list_to_mix_sep_spec_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         sep_model=None,
                         target_source=None):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.
    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.
    return : numpy.array( numpy.array( float ) )
        training dataset (when generate the validation data, this function is not used.)
        * dataset.shape = (total_dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 loop of file_to_vectorarray
    for idx in tqdm(range(len(file_list)), desc=msg):
        target_type = os.path.split(os.path.split(os.path.split(file_list[idx])[0])[0])[1]

        sr, mixture_y, active_label_sources = train_file_to_mixture_wav_label(file_list[idx])
            
        active_labels = torch.stack([active_label_sources[src] for src in machine_types])
        _, time = sep_model(torch.Tensor(mixture_y).unsqueeze(0).cuda(), active_labels.unsqueeze(0).cuda())
        # [src, b, ch, time]
        if target_source is not None:
            target_idx = machine_types.index(target_source)
        else:
            target_idx = machine_types.index(target_type)
        ys = time[target_idx, 0, 0, :].detach().cpu().numpy()
        
        vector_array = wav_to_spec_vector_array(sr, ys,
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)

        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)

        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset


class AEDataset(torch.utils.data.Dataset):
    def __init__(self, 
            sep_model, 
            file_list,
            param,
            target_source=None,
            ):
        self.sep_model = sep_model
        self.file_list = file_list
        self.target_source = target_source

        self.data_vector = train_list_to_mix_sep_spec_vector_array(self.file_list,
                                            msg="generate train_dataset",
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"],
                                            sep_model = self.sep_model,
                                            target_source=target_source)
        
    
    def __getitem__(self, index):
        return torch.Tensor(self.data_vector[index, :])
    
    def __len__(self):
        return self.data_vector.shape[0]


def dataset_generator(target_dir,
                      normal_dir_name="normal",
                      abnormal_dir_name="abnormal",
                      ext="wav"):
    """
    target_dir : str
        base directory path of the dataset
    normal_dir_name : str (default="normal")
        directory name the normal data located in
    abnormal_dir_name : str (default="abnormal")
        directory name the abnormal data located in
    ext : str (default="wav")
        filename extension of audio files 
    return : 
        train_data : numpy.array( numpy.array( float ) )
            training dataset
            * dataset.shape = (total_dataset_size, feature_vector_length)
        train_files : list [ str ]
            file list for training
        train_labels : list [ boolean ] 
            label info. list for training
            * normal/abnormal = 0/1
        eval_files : list [ str ]
            file list for evaluation
        eval_labels : list [ boolean ] 
            label info. list for evaluation
            * normal/abnormal = 0/1
    """

    logger.info("target_dir : {}".format(target_dir))

    # 01 normal list generate
    normal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                 normal_dir_name=normal_dir_name,
                                                                ext=ext))))
    normal_len = [len(glob.glob(
        os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir.replace(S1, mt),
                                                                 normal_dir_name=normal_dir_name,
                                                                 ext=ext)))) for mt in machine_types]   #dataset 중에서 가장 짧은 것 기준
    normal_len = min(min(normal_len), len(normal_files))
    normal_files = normal_files[:normal_len]


    normal_labels = numpy.zeros(len(normal_files))
    if len(normal_files) == 0:
        logger.exception("no_wav_data!!")


    # 02 abnormal list generate
    abnormal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                   abnormal_dir_name=abnormal_dir_name,
                                                                   ext=ext))))
    abnormal_files.extend(sorted(glob.glob(
        os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir.replace(S1, S2),
                                                                   abnormal_dir_name=abnormal_dir_name,
                                                                 ext=ext)))))                                               
    abnormal_labels = numpy.ones(len(abnormal_files))
    if len(abnormal_files) == 0:
        logger.exception("no_wav_data!!")

    # 03 separate train & eval
    train_files = normal_files[num_eval_normal:]
    train_labels = normal_labels[num_eval_normal:]
    eval_normal_files = sum([[fan_file.replace(S1, machine_type) for fan_file in normal_files[:num_eval_normal]] for machine_type in machine_types], [])
    eval_files = numpy.concatenate((eval_normal_files, abnormal_files), axis=0)
    eval_labels = numpy.concatenate((normal_labels[:num_eval_normal], normal_labels[:num_eval_normal], abnormal_labels), axis=0)  ##TODO 
    logger.info("train_file num : {num}".format(num=len(train_files)))
    logger.info("eval_file  num : {num}".format(num=len(eval_files)))

    return train_files, train_labels, eval_files, eval_labels


def fix_seed(seed: int = 42):
    random.seed(seed) # random
    numpy.random.seed(seed) # numpy
    os.environ["PYTHONHASHSEED"] = str(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

########################################################################



########################################################################
# main
########################################################################
if __name__ == "__main__":

    # load parameter yaml
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)
    
    # set random seed fixed
    fix_seed(param['seed'])
    
    # make output directory
    os.makedirs(param["pickle_directory"], exist_ok=True)
    os.makedirs(param["model_directory"], exist_ok=True)
    os.makedirs(param["result_directory"], exist_ok=True)


    # load base_directory list
    dirs = sorted(glob.glob(os.path.abspath("{base}/6dB/{machine}/{source}".format(base=param["base_directory"], machine = MACHINE, source = S1))))
    # setup the result
    result_file = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
    results = {}
    
    # loop of the base directory
    for dir_idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{num}/{total}] {dirname}".format(dirname=target_dir, num=dir_idx + 1, total=len(dirs)))
        print(target_dir)

        # dataset param        
        db = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
        machine_type = 'mix'
        machine_id = os.path.split(target_dir)[1] 
        # target_dir = target_dir.replace('fan', '*')

        # setup path
        evaluation_result = {}
        train_pickle = "{pickle}/src_train_{machine_type}_{machine_id}_{db}.pickle".format(pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id, db=db)
        eval_files_pickle = "{pickle}/src_eval_files_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        eval_labels_pickle = "{pickle}/src_eval_labels_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        model_file = "{model}/src_model_{machine_type}_{machine_id}_{db}.pth".format(model=param["model_directory"],
                                                                                  machine_type=machine_type,
                                                                                  machine_id=machine_id,
                                                                                  db=db)
        history_img = "{model}/src_history_{machine_type}_{machine_id}_{db}.png".format(model=param["model_directory"],
                                                                                    machine_type=machine_type,
                                                                                    machine_id=machine_id,
                                                                                    db=db)
        evaluation_result_key = "{machine_type}_{machine_id}_{db}".format(machine_type=machine_type,
                                                                          machine_id=machine_id,
                                                                          db=db)
   
        ae_path = f'{ae_path_base}/{MACHINE}'
        os.makedirs(ae_path, exist_ok=True)

        sep_model = xumx_model(xumx_model_path)
        sep_model = sep_model.cuda()
        sep_model.eval()


        # dataset generator
        print("============== DATASET_GENERATOR ==============")
        if os.path.exists(train_pickle) and os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle):
            train_files, train_labels = load_pickle(train_pickle)
            eval_files = load_pickle(eval_files_pickle)
            eval_labels = load_pickle(eval_labels_pickle)
        else:
            train_files, train_labels, eval_files, eval_labels = dataset_generator(target_dir)
            save_pickle(train_pickle, (train_files, train_labels))
            save_pickle(eval_files_pickle, eval_files)
            save_pickle(eval_labels_pickle, eval_labels)
        
        model = {}
        for target_type in machine_types:
            train_dataset = AEDataset(sep_model, train_files, param, target_source=target_type)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=param["fit"]["batch_size"], shuffle=True,
            )
            
            os.makedirs(os.path.join(ae_path, target_type), exist_ok = True)
            ae_path_target = os.path.join(ae_path, target_type, FILE)

            # model training
            print("============== MODEL TRAINING ==============")
            dim_input = train_dataset.data_vector.shape[1]
            model[target_type] = TorchModel(dim_input).cuda()
            #model[target_type](torch.load(model_file))
            optimizer = torch.optim.Adam(model[target_type].parameters(), lr=1.0e-2)
            loss_fn = nn.MSELoss()

            for epoch in range(param["fit"]["epochs"]):
                losses = []
                for batch in train_loader:
                    batch = batch.cuda()
                    pred = model[target_type](batch)
                    loss = loss_fn(pred, batch)
    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                if epoch % 10 == 0:
                    print(f"epoch {epoch}: loss {sum(losses) / len(losses)}")
            torch.save(model[target_type].state_dict(), ae_path_target)
            model[target_type].eval()
               
        # evaluation
        print("============== EVALUATION ==============")
        y_pred_mean = numpy.array([0. for k in eval_labels])
        y_pred_max = numpy.array([0. for k in eval_labels])
        y_true = numpy.array(eval_labels)
        sdr_pred_normal = {mt: [] for mt in machine_types}
        sdr_pred_abnormal = {mt: [] for mt in machine_types}

        eval_types = {mt: [] for mt in machine_types}
        
        overlap_log = []
        
        # for file in eval_files:
        #     print(file)
            
        for num, file_name in tqdm(enumerate(eval_files), total=len(eval_files)):
            machine_type = os.path.split(os.path.split(os.path.split(file_name)[0])[0])[1]
            target_idx = machine_types.index(machine_type)  
            
            sr, mixture_y, y_raw, active_label_sources = eval_file_to_mixture_wav_label(file_name)
            overlap_ratio = get_overlap_ratio(active_label_sources[machine_types[0]], active_label_sources[machine_types[1]])
            
            active_labels = torch.stack([active_label_sources[src] for src in machine_types])
            _, time = sep_model(torch.Tensor(mixture_y).unsqueeze(0).cuda(), active_labels.unsqueeze(0).cuda())
            # [src, b, ch, time]
            ys = time[target_idx, 0, 0, :].detach().cpu().numpy()
            
            data = wav_to_spec_vector_array(sr, ys,
                                        n_mels=param["feature"]["n_mels"],
                                        frames=param["feature"]["frames"],
                                        n_fft=param["feature"]["n_fft"],
                                        hop_length=param["feature"]["hop_length"],
                                        power=param["feature"]["power"])
            
            data = torch.Tensor(data).cuda()
            error = torch.mean(((data - model[machine_type](data)) ** 2), dim=1)

            sep_sdr, _, _, _ = museval.evaluate(numpy.expand_dims(y_raw[machine_type][0, :ys.shape[0]], axis=(0,2)), 
                                        numpy.expand_dims(ys, axis=(0,2)))

            y_pred_mean[num] = torch.mean(error).detach().cpu().numpy()
            y_pred_max[num] = torch.max(error).detach().cpu().numpy()
            
            overlap_log.append([
                    'normal' if num < num_eval_normal * 2 else 'abnormal',
                    machine_type,
                    numpy.mean(sep_sdr),
                    torch.mean(error).detach().cpu().item(), 
                    torch.max(error).detach().cpu().item(), 
                    overlap_ratio.item()
                    ])
            
            eval_types[machine_type].append(num)

            if num < num_eval_normal * 2: # normal file
                sdr_pred_normal[machine_type].append(numpy.mean(sep_sdr))
            else: # abnormal file
                sdr_pred_abnormal[machine_type].append(numpy.mean(sep_sdr))

        with open('overlap_sep.pkl', 'wb') as f:
            pickle.dump(overlap_log, f)

        mean_scores = []
        max_scores = []
        anomaly_detect_score = {}

        for machine_type in machine_types:
            mean_score = metrics.roc_auc_score(y_true[eval_types[machine_type]], y_pred_mean[eval_types[machine_type]])
            max_score = metrics.roc_auc_score(y_true[eval_types[machine_type]], y_pred_max[eval_types[machine_type]])
            logger.info("AUC_mean_{} : {}".format(machine_type, mean_score))
            logger.info("AUC_max_{} : {}".format(machine_type, max_score))
            evaluation_result["AUC_mean_{}".format(machine_type)] = float(mean_score)
            evaluation_result["AUC_max_{}".format(machine_type)] = float(max_score)
            mean_scores.append(mean_score)
            max_scores.append(max_score)
            logger.info("SDR_normal_{} : {}".format(machine_type, sum(sdr_pred_normal[machine_type])/len(sdr_pred_normal[machine_type])))
            logger.info("SDR_abnormal_{} : {}".format(machine_type, sum(sdr_pred_abnormal[machine_type])/len(sdr_pred_abnormal[machine_type])))
            evaluation_result["SDR_normal_{}".format(machine_type)] = float(sum(sdr_pred_normal[machine_type])/len(sdr_pred_normal[machine_type]))
            evaluation_result["SDR_abnormal_{}".format(machine_type)] = float(sum(sdr_pred_abnormal[machine_type])/len(sdr_pred_abnormal[machine_type]))
        
        mean_score = sum(mean_scores) / len(mean_scores)
        max_score = sum(max_scores) / len(max_scores)
        logger.info("AUC_mean : {}".format(mean_score))
        logger.info("AUC_max : {}".format(max_score))
        evaluation_result["AUC_mean"] = float(mean_score)
        evaluation_result["AUC_max"] = float(max_score)
        results[evaluation_result_key] = evaluation_result
        print("===========================")

    # output results
    print("\n===========================")
    logger.info("all results -> {}".format(result_file))
    with open(result_file, "w") as f:
        f.write(yaml.dump(results, default_flow_style=False))
    print("===========================")