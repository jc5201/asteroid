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


MACHINE = 'slider'
num_eval_normal = 250


########################################################################
# feature extractor
########################################################################

def generate_spec_label(y):
    # np, [c, t]
    channels = y.shape[0]
    rms_fig = librosa.feature.rms(y=y) 
    #[c, 1, 313]

    frames = 5
    rms_trim = torch.stack([torch.tensor(rms_fig[:, 0, i:i+rms_fig.shape[2]-frames+1]) for i in range(frames)], dim=2)
    #[c, 313-4, 5]

    if MACHINE == 'valve':
        k = int(rms_fig.shape[2]*0.8)
        min_threshold, _ = torch.kthvalue(torch.tensor(rms_fig)[0, 0, :], k)
    else:
        min_threshold = (torch.max(torch.tensor(rms_fig)) + torch.min(torch.tensor(rms_fig)))/2

    label = (rms_trim > min_threshold).type(torch.float) 
    #[c, 313-4, 5]
    return label

def eval_file_to_wav_label(filename):
    machine_type = os.path.split(os.path.split(os.path.split(filename)[0])[0])[1]
    ys = 0

    src_filename = filename
    sr, y = demux_wav(src_filename)
    ys = ys + y
    active_spec_label = generate_spec_label(numpy.expand_dims(y, axis=0))
    
    return sr, ys, active_spec_label

def list_to_spec_vector_array(file_list,
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

        vector_array = file_to_spec_vector_array(file_list[idx],
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
            file_list,
            param,
            target_source=None,
            ):
        self.file_list = file_list
        self.target_source = target_source

        self.data_vector = list_to_spec_vector_array(self.file_list,
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
    normal_labels = numpy.zeros(len(normal_files))
    if len(normal_files) == 0:
        logger.exception("no_wav_data!!")

    # 02 abnormal list generate
    abnormal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                   abnormal_dir_name=abnormal_dir_name,
                                                                   ext=ext))))                              
    abnormal_labels = numpy.ones(len(abnormal_files))
    if len(abnormal_files) == 0:
        logger.exception("no_wav_data!!")

    # 03 separate train & eval
    train_files = normal_files[len(abnormal_files):]
    train_labels = normal_labels[len(abnormal_files):]
    eval_files = numpy.concatenate((normal_files[:len(abnormal_files)], abnormal_files), axis=0)
    eval_labels = numpy.concatenate((normal_labels[:len(abnormal_files)], abnormal_labels), axis=0)
    logger.info("train_file num : {num}".format(num=len(train_files)))
    logger.info("eval_file  num : {num}".format(num=len(eval_files)))

    return train_files, train_labels, eval_files, eval_labels


########################################################################


########################################################################
# main
########################################################################
if __name__ == "__main__":

    # load parameter yaml
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)


    fix_seed(param['seed'])
    
    # make output directory
    os.makedirs(param["pickle_directory"], exist_ok=True)
    os.makedirs(param["model_directory"], exist_ok=True)
    os.makedirs(param["result_directory"], exist_ok=True)

    # load base_directory list
    dirs = sorted(glob.glob(os.path.abspath("{base}/6dB/{machine}/*".format(base=param["base_directory"], machine=MACHINE))))
    dirs = dirs[2:]
    print(dirs)

    # setup the result
    result_file = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
    results = {}

    # loop of the base directory
    for dir_idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{num}/{total}] {dirname}".format(dirname=target_dir, num=dir_idx + 1, total=len(dirs)))

        # dataset param        
        db = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
        machine_type = os.path.split(os.path.split(target_dir)[0])[1]
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
        model_file = "{model}/model_{machine_type}_{machine_id}_{db}.pth".format(model=param["model_directory"],
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
        #     train_files = load_pickle(train_pickle)
        #     eval_files = load_pickle(eval_files_pickle)
        #     eval_labels = load_pickle(eval_labels_pickle)
        # else:
        train_files, train_labels, eval_files, eval_labels  = dataset_generator(target_dir)

        save_pickle(train_pickle, train_files)
        save_pickle(eval_files_pickle, eval_files)
        save_pickle(eval_labels_pickle, eval_labels)

        if dir_idx == 0:
            train_dataset = AEDataset(train_files, param)
            eval_files_ = eval_files
            eval_labels_ = eval_labels
            eval_num = len(eval_labels)
        else:
            train_dataset_ = AEDataset(train_files, param)
            train_dataset.data_vector = numpy.concatenate((train_dataset.data_vector, train_dataset_.data_vector), axis = 0)
            eval_files = numpy.concatenate((eval_files_, eval_files), axis = 0)
            eval_labels = numpy.concatenate((eval_labels_, eval_labels), axis = 0)
           
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=param["fit"]["batch_size"], shuffle=True,
    )
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
    y_pred_mean = [0. for k in eval_labels]
    y_pred_max = [0. for k in eval_labels]
    y_pred_mask = [0. for k in eval_labels]
    y_true = eval_labels
    
    for num, file_name in tqdm(enumerate(eval_files), total=len(eval_files)):
        
        sr, ys, active_spec_label = eval_file_to_wav_label(file_name)
        # active_label [channel, time]
        data = wav_to_spec_vector_array(sr, ys,
                                    n_mels=param["feature"]["n_mels"],
                                    frames=param["feature"]["frames"],
                                    n_fft=param["feature"]["n_fft"],
                                    hop_length=param["feature"]["hop_length"],
                                    power=param["feature"]["power"])
        
        n_mels = param["feature"]["n_mels"]
        frames = param["feature"]["frames"]
        # [1, 309, 5] -> [309, 5*n_mels]
        active_spec_label = active_spec_label.cuda().unsqueeze(3).repeat(1, 1, 1, n_mels).reshape(1, 309, frames * n_mels).squeeze(0)

        data = torch.Tensor(data).cuda()
        error = torch.mean((data - model(data)) ** 2, dim=1)
        error_mask = torch.mean(((data - model(data)) * active_spec_label) ** 2, dim=1)
        y_pred_mean[num] = torch.mean(error).detach().cpu().numpy()
        y_pred_max[num] = torch.max(error).detach().cpu().numpy()
        y_pred_mask[num] = torch.mean(error_mask).detach().cpu().numpy()
        
        
    # save model
    torch.save(model.state_dict(), model_file)
   
    mean_score1 = metrics.roc_auc_score(y_true[-eval_num:], y_pred_mean[-eval_num:])
    max_score1 = metrics.roc_auc_score(y_true[-eval_num:], y_pred_max[-eval_num:])
    mask_score1 = metrics.roc_auc_score(y_true[-eval_num:], y_pred_mask[-eval_num:])
    
    mean_score2 = metrics.roc_auc_score(y_true[:-eval_num], y_pred_mean[:-eval_num],)
    max_score2= metrics.roc_auc_score(y_true[:-eval_num], y_pred_max[:-eval_num],)
    mask_score2 = metrics.roc_auc_score(y_true[:-eval_num], y_pred_mask[:-eval_num],)
    
    logger.info("AUC_mean_{} : {}".format('id_04', mean_score1))
    logger.info("AUC_max_{} : {}".format('id_04', max_score1))
    logger.info("AUC_mask_{} : {}".format('id_04',mask_score1))
    
    logger.info("AUC_mean_{} : {}".format('id_06', mean_score2))
    logger.info("AUC_max_{} : {}".format('id_06', max_score2))
    logger.info("AUC_mask_{} : {}".format('id_06',mask_score2))
    
    evaluation_result["AUC_mean_id04"] = float(mean_score1)
    evaluation_result["AUC_max_id04"] = float(max_score1)
    evaluation_result["AUC_mask_id04"] = float(mask_score1)
    
    evaluation_result["AUC_mean_id06"] = float(mean_score2)
    evaluation_result["AUC_max_id06"] = float(max_score2)
    evaluation_result["AUC_mask_id06"] = float(mask_score2)
    
    mean_score = (mean_score1 + mean_score2)/2
    max_score = (max_score1 + max_score2)/2
    mask_score = (mask_score1 + mask_score2)/2
    
    evaluation_result["AUC_mean"] = float(mean_score)
    evaluation_result["AUC_max"] = float(max_score)
    evaluation_result["AUC_mask"] = float(mask_score)
    results[evaluation_result_key] = evaluation_result
    print("===========================")
    
# output results
print("\n===========================")
logger.info("all results -> {}".format(result_file))
with open(result_file, "w") as f:
    f.write(yaml.dump(results, default_flow_style=False))
print("===========================")