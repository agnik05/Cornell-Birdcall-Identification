import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sklearn.metrics as sm
import sklearn.model_selection as sms

import src.datasets as datasets
from src.transforms import (get_waveform_transforms,
                            get_spectrogram_transforms)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_metadata(config):
    metadata_config = config['metadata']
    train_df = pd.read_csv(metadata_config['csv_file'])

    tmp_list = []
    audio_dirs = metadata_config['audio_dirs']
    for audio_dir in audio_dirs:
        audio_dir = Path(audio_dir)
        if not audio_dir.exists():
            continue
        for ebird_dir in audio_dir.iterdir():
            if ebird_dir.is_file():
                continue
            for audio in ebird_dir.iterdir():
                tmp_list.append([ebird_dir.name, audio.name, audio.as_posix()])

    resampled_df = pd.DataFrame(tmp_list, columns=['ebird_code', 'resampled_filename', 'file_path'])
    df = pd.merge(train_df, resampled_df, on=['ebird_code', 'resampled_filename'], how='inner')
    return df

def get_split(config):
    split_config = config['split']
    split_name = split_config['name']
    return sms.__getattribute__(split_name)(**split_config['params'])

def get_criterion(config):
    criterion_config = config['criterion']
    criterion_name = criterion_config['name']
    return nn.__getattribute__(criterion_name)(**criterion_config['params'])

def get_optimizer(config, model):
    optimizer_config = config['optimizer']
    optimizer_name = optimizer_config['name']
    return optim.__getattribute__(optimizer_name)(
        model.parameters(),
        **optimizer_config['params']
    )

def get_scheduler(config, optimizer):
    scheduler_config = config['scheduler']
    scheduler_name = scheduler_config['name']
    return optim.lr_scheduler.__getattribute__(scheduler_name)(
        optimizer,
        **scheduler_config['params']
    )

def get_loaders(config, train_files, valid_files):
    loader_config = config['loader']
    dataset_config = config['dataset']

    waveform_transforms = get_waveform_transforms()
    spectrogram_transforms = get_spectrogram_transforms()
    train_dataset = datasets.CornellDataset(train_files,
                                            dataset_config,
                                            waveform_transforms,
                                            spectrogram_transforms)
    valid_dataset = datasets.CornellDataset(valid_files,
                                            dataset_config,
                                            waveform_transforms,
                                            spectrogram_transforms)
    train_loader = DataLoader(train_dataset, **loader_config['train'])
    valid_loader = DataLoader(valid_dataset, **loader_config['valid'])
    return train_loader, valid_loader

def get_score(config, y_pred, y_true, threshold):
    metric_config = config['metric']
    metric_name = metric_config.get('name')
    score = sm.__getattribute__(metric_name)(
        y_pred > threshold,
        y_true,
        **metric_config['params']
    )
    return score
