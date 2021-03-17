config = {

    'metadata': {
        'csv_file': 'D:/data/cornell-birdcall-identification/birdsong-resampled-train-audio-00/train_mod.csv',
        'audio_dirs': ['D:/data/cornell-birdcall-identification/birdsong-resampled-train-audio-00',
                       'D:/data/cornell-birdcall-identification/birdsong-resampled-train-audio-01',
                       'D:/data/cornell-birdcall-identification/birdsong-resampled-train-audio-02',
                       'D:/data/cornell-birdcall-identification/birdsong-resampled-train-audio-03',
                       'D:/data/cornell-birdcall-identification/birdsong-resampled-train-audio-04']
    },

    'split': {
        'name': 'StratifiedKFold',
        'params': {
            'n_splits': 5,
            'random_state': 2020,
            'shuffle': True
        }
    },

    'model': {
        'name': 'resnext50',
        'params': {
            'pretrained': True,
            'num_classes': 264
        }
    },

    'criterion': {
        'name': 'BCELoss',
        'params': {}
    },

    'optimizer': {
        'name': 'Adam',
        'params': {
            'lr': 0.001
        }
    },

    'scheduler': {
        'name': 'CosineAnnealingLR',
        'params': {
            'T_max': 10
        }
    },

    'dataset': {
        'img_size': 224,
        'audio_params': {
            'target_duration': 5,
            'sampling_rate': 32000
        },
        'spectrogram_params': {
            'n_mels': 128,
            'fmin': 20,
            'fmax': 16000
        }
    },

    'loader': {
        'train': {
            'batch_size': 16,
            'shuffle': True,
            'num_workers': 6,
            'pin_memory': True,
            'drop_last': True
        },
        'valid': {
            'batch_size': 32,
            'shuffle': False,
            'num_workers': 6,
            'pin_memory': True,
            'drop_last': False
        }
    },

    'metric': {
        'name': 'f1_score',
        'params': {
            'average': 'micro'
        }
    }
}
