from src.labels import BIRD_CODE

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset


def read_audio(audio_file, audio_duration, target_duration=5, target_sampling_rate=32000):
    if audio_duration > target_duration:
        offset = np.random.choice(audio_duration - target_duration)
        y, sr = sf.read(audio_file,
                        start=offset,
                        frames=target_duration * target_sampling_rate)
    else:
        y, sr = sf.read(audio_file,
                        frames=target_duration * target_sampling_rate,
                        fill_value=0)
    y = y.astype(np.float32)
    return y, sr


class CornellDataset(Dataset):

    def __init__(self, file_list, waveform_transforms=None):
        self.file_list = file_list
        self.waveform_transforms = waveform_transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, ebird_code, duration = self.file_list[idx]
        y, _ = utils.read_audio(file_path, duration)
        label = np.zeros(len(BIRD_CODE), dtype='f')
        label[BIRD_CODE[ebird_code]] = 1
        return y, label