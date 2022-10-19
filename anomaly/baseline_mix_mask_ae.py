#!/usr/bin/env python
########################################################################
# import default python-library
########################################################################
import os
import sys
import glob

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

from baseline_mix import *
########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.3"
########################################################################

S1 = 'id_00'
S2 = 'id_02'
MACHINE = 'slider'
machine_types = [S1, S2]
num_eval_normal = 250

########################################################################
# feature extractor
########################################################################

def train_file_to_mixture_wav(filename):
    machine_type = os.path.split(os.path.split(os.path.split(filename)[0])[0])[1]
    ys = 0
    for machine in machine_types:
        src_filename = filename.replace(machine_type, machine)
        sr, y = demux_wav(src_filename)
        ys = ys + y

    _, active_spec_label = generate_label(numpy.expand_dims(ys, axis=0), MACHINE)

    return sr, ys, active_spec_label
    
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
    active_spec_label_sources = {}
    for normal_type in machine_types:
        if normal_type == machine_type:
            src_filename = filename
        else:
            src_filename = filename.replace(machine_type, normal_type).replace('abnormal', 'normal')
        sr, y = demux_wav(src_filename)
        ys = ys + y
        label, spec_label = generate_label(np.expand_dims(y, axis=0), MACHINE)
        active_label_sources[normal_type] = label
        active_spec_label_sources[normal_type] = spec_label
        gt_wav[normal_type] = y
    
    return sr, ys, gt_wav, active_label_sources, active_spec_label_sources

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

        sr, ys, spec_label = train_file_to_mixture_wav(file_list[idx])
        
        spec_label = spec_label.unsqueeze(3).repeat(1, 1, 1, n_mels).reshape(1, 309, frames * n_mels).squeeze(0).numpy()

        vector_array = wav_to_spec_vector_array(sr, ys,
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power,
                                            spec_mask=spec_label)

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
        #     train_files = load_pickle(train_pickle)
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
        y_pred_mean = numpy.array([0. for k in eval_labels])
        y_pred_max = numpy.array([0. for k in eval_labels])
        y_pred_mask = numpy.array([0. for k in eval_labels])
        y_true = numpy.array(eval_labels)

        eval_types = {mt: [] for mt in machine_types}
    
        for num, file_name in tqdm(enumerate(eval_files), total=len(eval_files)):
            machine_type = os.path.split(os.path.split(os.path.split(file_name)[0])[0])[1]

            sr, ys, y_raw, active_label_sources, active_spec_label_sources = eval_file_to_mixture_wav_label(file_name)
            # overlap_ratio = get_overlap_ratio(active_label_sources[machine_types[0]], active_label_sources[machine_types[1]])

            n_mels = param["feature"]["n_mels"]
            frames = param["feature"]["frames"]
            # [1, 309, 5] -> [309, 5*n_mels]
            active_spec_label = active_spec_label_sources[machine_type].cuda().unsqueeze(3).repeat(1, 1, 1, n_mels).reshape(1, 309, frames * n_mels).squeeze(0)
            
            
            data = wav_to_spec_vector_array(sr, ys,
                                        n_mels=param["feature"]["n_mels"],
                                        frames=param["feature"]["frames"],
                                        n_fft=param["feature"]["n_fft"],
                                        hop_length=param["feature"]["hop_length"],
                                        power=param["feature"]["power"],
                                        spec_mask=active_spec_label.cpu().numpy())


            data = torch.Tensor(data).cuda()
            error = torch.mean(((data - model(data)) ** 2), dim=1)
            error_mask = torch.mean(((data - model(data)) * active_spec_label) ** 2, dim=1)
            y_pred_mean[num] = torch.mean(error).detach().cpu().numpy()
            y_pred_max[num] = torch.max(error).detach().cpu().numpy()
            y_pred_mask[num] = torch.mean(error_mask).detach().cpu().numpy()

            if num < num_eval_normal:
                for mt in machine_types:
                    eval_types[mt].append(num)
            else:
                eval_types[machine_type].append(num)

        mean_scores = []
        max_scores = []
        mask_scores = []
        for machine_type in machine_types:
            mean_score = metrics.roc_auc_score(y_true[eval_types[machine_type]], y_pred_mean[eval_types[machine_type]])
            max_score = metrics.roc_auc_score(y_true[eval_types[machine_type]], y_pred_max[eval_types[machine_type]])
            mask_score = metrics.roc_auc_score(y_true[eval_types[machine_type]], y_pred_mask[eval_types[machine_type]])
            logger.info("AUC_mean_{} : {}".format(machine_type, mean_score))
            logger.info("AUC_max_{} : {}".format(machine_type, max_score))
            logger.info("AUC_mask_{} : {}".format(machine_type, mask_score))
            evaluation_result["AUC_mean_{}".format(machine_type)] = float(mean_score)
            evaluation_result["AUC_max_{}".format(machine_type)] = float(max_score)
            evaluation_result["AUC_mask_{}".format(machine_type)] = float(mask_score)
            mean_scores.append(mean_score)
            max_scores.append(max_score)
            mask_scores.append(mask_score)
        mean_score = sum(mean_scores) / len(mean_scores)
        max_score = sum(max_scores) / len(max_scores)
        mask_score = sum(mask_scores) / len(mask_scores)
        logger.info("AUC_mean : {}".format(mean_score))
        logger.info("AUC_max : {}".format(max_score))
        logger.info("AUC_mask : {}".format(mask_score))
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
########################################################################