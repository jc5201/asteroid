#!/usr/bin/env python
"""
 @file   baseline.py
 @brief  Baseline code of simple AE-based anomaly detection used experiment in [1].
 @author Ryo Tanabe and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2019 Hitachi, Ltd. All right reserved.
 [1] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection," arXiv preprint arXiv:1909.09347, 2019.
"""
########################################################################
# import default python-library
########################################################################
import os
import sys
import glob
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
import numpy as np
import librosa
import librosa.core
import librosa.feature
import yaml
# from import
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.nn as nn

from utils import *
from model import TorchModel
########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.3"
########################################################################

S1 = 'id_00'
S2 = 'id_02'
MACHINE = 'valve'
machine_types = [S1, S2]
num_eval_normal = 250

########################################################################
# feature extractor
########################################################################

def generate_label(y):
    rms_fig = librosa.feature.rms(y=y)
    rms_tensor = torch.tensor(rms_fig).reshape(1, -1, 1)
    rms_trim = rms_tensor.expand(-1, -1, 512).reshape(1, -1)[:, :160000]

    if MACHINE == 'valve':
        k = int(y.shape[1]*0.8)
        min_threshold, _ = torch.kthvalue(rms_trim, k)
    else:
        min_threshold = (torch.max(rms_trim) + torch.min(rms_trim))/2
    label = (rms_trim > min_threshold).type(torch.float)
    label = label.expand(y.shape[0], -1)
    return label

def train_file_to_mixture_wav(filename):
    machine_type = os.path.split(os.path.split(os.path.split(filename)[0])[0])[1]
    ys = 0
    for machine in machine_types:
        src_filename = filename.replace(machine_type, machine)
        sr, y = demux_wav(src_filename)
        ys = ys + y

    return sr, ys
    
def eval_file_to_mixture_wav(filename):
    machine_type = os.path.split(os.path.split(os.path.split(filename)[0])[0])[1]
    ys = 0
    for normal_type in machine_types:
        if normal_type == machine_type:
            src_filename = filename
        else:
            src_filename = filename.replace(machine_type, normal_type).replace('abnormal', 'normal')
        sr, y = demux_wav(src_filename)
        ys = ys + y
    
    return sr, ys

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
        sr, y = demux_wav(src_filename)
        ys = ys + y
        active_label_sources[normal_type] = generate_label(np.expand_dims(y, axis=0))
        gt_wav[normal_type] = y
    
    return sr, ys, gt_wav, active_label_sources

def get_overlap_ratio(signal1, signal2):
    return torch.sum(torch.logical_and(signal1, signal2)) / torch.sum(torch.logical_or(signal1, signal2))

def train_list_to_mixture_spec_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
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

        sr, ys = train_file_to_mixture_wav(file_list[idx])
        
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


class AEDatasetMix(torch.utils.data.Dataset):
    def __init__(self, 
            file_list,
            param,
            target_source=None,
            ):
        self.file_list = file_list
        self.target_source = target_source

        self.data_vector = train_list_to_mixture_spec_vector_array(self.file_list,
                                            msg="generate train_dataset",
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"],
                                            )        
    
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
        os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir.replace('id_00', mt),
                                                                 normal_dir_name=normal_dir_name,
                                                                 ext=ext)))) for mt in machine_types]
    normal_len = min(min(normal_len), len(normal_files))
    normal_files = normal_files[:normal_len]

    normal_labels = numpy.zeros(len(normal_files))
    if len(normal_files) == 0:
        logger.exception("no_wav_data!!")

    # 02 abnormal list generate
    abnormal_files = []
    for machine_type in machine_types:
        abnormal_files.extend(sorted(glob.glob(
            os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir.replace('id_00', machine_type),
                                                                 abnormal_dir_name=abnormal_dir_name,
                                                                 ext=ext)))))         

    abnormal_labels = numpy.ones(len(abnormal_files))
    if len(abnormal_files) == 0:
        logger.exception("no_wav_data!!")

    # 03 separate train & eval
    train_files = normal_files[num_eval_normal:]
    train_labels = normal_labels[num_eval_normal:]
    eval_files = numpy.concatenate((normal_files[:num_eval_normal], abnormal_files), axis=0)
    eval_labels = numpy.concatenate((normal_labels[:num_eval_normal], abnormal_labels), axis=0)
    logger.info("train_file num : {num}".format(num=len(train_files)))
    logger.info("eval_file  num : {num}".format(num=len(eval_files)))

    return train_files, train_labels, eval_files, eval_labels


########################################################################
# main
########################################################################
if __name__ == "__main__":

    # load parameter yaml
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)

    # make output directory
    os.makedirs(param["pickle_directory"], exist_ok=True)
    os.makedirs(param["model_directory"], exist_ok=True)
    os.makedirs(param["result_directory"], exist_ok=True)

    # load base_directory list
    dirs = sorted(glob.glob(os.path.abspath("{base}/6dB/{machine}/{source}".format(base=param["base_directory"], machine = MACHINE, source = S1))))  # {base}/0dB/fan/id_00/normal/00000000.wav

    # setup the result
    result_file = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
    results = {}

    # loop of the base directory
    for dir_idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{num}/{total}] {dirname}".format(dirname=target_dir, num=dir_idx + 1, total=len(dirs)))

        # dataset param        
        db = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
        machine_type = 'mix'
        machine_id = os.path.split(target_dir)[1]

        # setup path
        evaluation_result = {}
        train_pickle = "{pickle}/train_{machine_type}_{machine_id}_{db}.pickle".format(pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id, db=db)
        eval_files_pickle = "{pickle}/eval_files_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        eval_labels_pickle = "{pickle}/eval_labels_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        model_file = "{model}/model_{machine_type}_{machine_id}_{db}.hdf5".format(model=param["model_directory"],
                                                                                  machine_type=machine_type,
                                                                                  machine_id=machine_id,
                                                                                  db=db)
        history_img = "{model}/history_{machine_type}_{machine_id}_{db}.png".format(model=param["model_directory"],
                                                                                    machine_type=machine_type,
                                                                                    machine_id=machine_id,
                                                                                    db=db)
        evaluation_result_key = "{machine_type}_{machine_id}_{db}".format(machine_type=machine_type,
                                                                          machine_id=machine_id,
                                                                          db=db)

        # dataset generator
        print("============== DATASET_GENERATOR ==============")
        # if os.path.exists(train_pickle) and os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle):
        #     train_files= load_pickle(train_pickle)
        #     eval_files = load_pickle(eval_files_pickle)
        #     eval_labels = load_pickle(eval_labels_pickle)
        # else:
        train_files, train_labels, eval_files, eval_labels = dataset_generator(target_dir)

        train_dataset = AEDatasetMix(train_files, param)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=param["fit"]["batch_size"], shuffle=True)

        save_pickle(train_pickle, train_files)
        save_pickle(eval_files_pickle, eval_files)
        save_pickle(eval_labels_pickle, eval_labels)

        # model training
        print("============== MODEL TRAINING ==============")
        dim_input = train_dataset.data_vector.shape[1]
        model = TorchModel(dim_input).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        loss_fn = nn.MSELoss()
 
        for epoch in range(param["fit"]["epochs"]):
            losses = []
            for batch in train_loader:
                batch = batch.cuda()
                pred = model(batch)
                loss = loss_fn(pred, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if epoch % 10 == 0:
                print(f"epoch {epoch}: loss {sum(losses) / len(losses)}")
        model.eval()

        # evaluation
        print("============== EVALUATION ==============")
        y_pred = numpy.array([0. for k in eval_labels])
        y_true = numpy.array(eval_labels)

        eval_types = {mt: [] for mt in machine_types}
        overlap_log = []
    
        for num, file_name in tqdm(enumerate(eval_files), total=len(eval_files)):
            machine_type = os.path.split(os.path.split(os.path.split(file_name)[0])[0])[1]

            sr, ys, y_raw, active_label_sources = eval_file_to_mixture_wav_label(file_name)
            overlap_ratio = get_overlap_ratio(active_label_sources[machine_types[0]], active_label_sources[machine_types[1]])

            data = wav_to_spec_vector_array(sr, ys,
                                        n_mels=param["feature"]["n_mels"],
                                        frames=param["feature"]["frames"],
                                        n_fft=param["feature"]["n_fft"],
                                        hop_length=param["feature"]["hop_length"],
                                        power=param["feature"]["power"])

            data = torch.Tensor(data).cuda()
            error = torch.mean(((data - model(data)) ** 2), dim=1)
            y_pred[num] = torch.mean(error).detach().cpu().numpy()

            overlap_log.append(['normal' if num < num_eval_normal else 'abnormal',
                    y_pred[num].item(0), 
                    torch.max(error).detach().cpu().item(), 
                    overlap_ratio])
            
            if num < num_eval_normal:
                for mt in machine_types:
                    eval_types[mt].append(num)
            else:
                eval_types[machine_type].append(num)

        with open('mix.pkl', 'wb') as f:
            pickle.dump(overlap_log, f)
        scores = []
        for machine_type in machine_types:
            score = metrics.roc_auc_score(y_true[eval_types[machine_type]], y_pred[eval_types[machine_type]])
            logger.info("AUC_{} : {}".format(machine_type, score))
            evaluation_result["AUC_{}".format(machine_type)] = float(score)
            scores.append(score)
        score = sum(scores) / len(scores)
        logger.info("AUC : {}".format(score))
        evaluation_result["AUC"] = float(score)
        results[evaluation_result_key] = evaluation_result
        print("===========================")

    # output results
    print("\n===========================")
    logger.info("all results -> {}".format(result_file))
    with open(result_file, "w") as f:
        f.write(yaml.dump(results, default_flow_style=False))
    print("===========================")
########################################################################