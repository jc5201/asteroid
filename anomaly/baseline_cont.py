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
import random
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
import torch.nn.functional as F

from utils import *
from model import TorchModel
########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.3"
########################################################################
MACHINE = 'valve'

########################################################################
# feature extractor
########################################################################

def generate_label(y):
    rms_fig = librosa.feature.rms(y)
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


def train_list_to_gt_spec_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         sep_model=None,
                         target_source=None):

    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 loop of file_to_vectorarray
    for idx in tqdm(range(len(file_list)), desc=msg):
        machine_type = os.path.split(os.path.split(os.path.split(file_list[idx])[0])[0])[1]
        
        sr, y = file_to_wav_stereo(file_list[idx])
        active_labels = generate_label(y)
        
        vector_array = wav_to_spec_vector_array(sr, y[0, :],
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)   # (309, 320) = (time, n_mels*frames)
        
           
        # convert active labels in wav domain to spectrogram domain
        Tb = vector_array.shape[0] + frames - 1
        active_labels_ = active_labels[:1, :].clone()
        control_spec = F.adaptive_max_pool1d(active_labels_, output_size=Tb) # (1, 313)
        control_spec_stack = numpy.zeros((vector_array.shape[0], frames), float) #(309, 5)
        for t in range(frames):
            control_spec_stack[:, t:(t + 1)] = control_spec[:, t: t + vector_array.shape[0]].T  # (309, 1) 
       
        # concat audio and activity labels
        vector_array = numpy.concatenate((vector_array, control_spec_stack), axis = 1)

        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims + frames), float)
          
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


class AEGTDataset(torch.utils.data.Dataset):
    def __init__(self,  
            file_list,
            param,
            target_source=None,
            ):
        
        self.file_list = file_list
        self.target_source = target_source

        self.data_vector = train_list_to_gt_spec_vector_array(self.file_list,
                                            msg="generate train_dataset",
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"],
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

    # set gpu
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"]= "4"  


    # load parameter yaml
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)

    # set random seed fixed
    fix_seed(param['seed'])

    # make output directory
    os.makedirs(param["pickle_directory"], exist_ok=True)
    os.makedirs(param["model_directory"], exist_ok=True)
    os.makedirs(param["result_directory"], exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # load base_directory list
    dirs = sorted(glob.glob(os.path.abspath("{base}/6dB/valve/*".format(base=param["base_directory"]))))
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
        if os.path.exists(train_pickle) and os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle):
            train_files = load_pickle(train_pickle)
            eval_files = load_pickle(eval_files_pickle)
            eval_labels = load_pickle(eval_labels_pickle)
        else:
            train_files, train_labels, eval_files, eval_labels = dataset_generator(target_dir)

            save_pickle(train_pickle, train_files)
            save_pickle(eval_files_pickle, eval_files)
            save_pickle(eval_labels_pickle, eval_labels)

        train_dataset = AEGTDataset(train_files, param)
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
                loss = loss_fn(pred[:, :320], batch[:, :320])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if epoch % 10 == 0:
                print(f"epoch {epoch}: loss {sum(losses) / len(losses)}")
        model.eval()

        # evaluation
        print("============== EVALUATION ==============")
        y_pred = [0. for k in eval_labels]
        y_true = eval_labels

        for num, file_name in tqdm(enumerate(eval_files), total=len(eval_files)):

            sr, ys = file_to_wav_stereo(file_name)
            active_labels = generate_label(ys)
            active_labels_ = active_labels[:1, :].clone()
            ys = ys[0, :]
            
            data = file_to_spec_vector_array(file_name,
                                        n_mels=param["feature"]["n_mels"],
                                        frames=param["feature"]["frames"],
                                        n_fft=param["feature"]["n_fft"],
                                        hop_length=param["feature"]["hop_length"],
                                        power=param["feature"]["power"])


            # convert active labels in wav domain to spectrogram domain
            frames = int(param["feature"]["frames"])
            Tb = data.shape[0] + frames - 1
            control_spec = F.adaptive_max_pool1d(active_labels_, output_size=Tb) # (1, 313)
            control_spec_stack = numpy.zeros((data.shape[0], frames), float) #(309, 5)
            for t in range(frames):
                control_spec_stack[:, t:(t + 1)] = control_spec[:, t: t + data.shape[0]].T  # (309, 1) 

            # concat audio and activity labels
            data = numpy.concatenate((data, control_spec_stack), axis = 1)
            data = torch.Tensor(data).cuda()
            error = torch.mean(((data[:, :320] - model(data)[:, :320]) ** 2), dim=1)
            y_pred[num] = torch.mean(error).detach().cpu().numpy()


        # save model
        torch.save(model.state_dict(), model_file)
        score = metrics.roc_auc_score(y_true, y_pred)
        logger.info("anomaly score abnormal : {}".format(str(numpy.array(y_pred)[y_true.astype(bool)])))
        logger.info("anomaly score normal : {}".format(str(numpy.array(y_pred)[numpy.logical_not(y_true)])))
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