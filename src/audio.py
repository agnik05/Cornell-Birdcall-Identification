import numpy as np
import soundfile as sf

def read_audio(audio_file, config, duration):
    if duration > config['target_duration']:
        offset = np.random.choice(duration - config['target_duration'])
        y, sr = sf.read(audio_file,
                        start=offset,
                        frames=config['target_duration'] * config['sampling_rate'])
    else:
        y, sr = sf.read(audio_file,
                        frames=config['target_duration'] * config['sampling_rate'],
                        fill_value=0)
    return y, sr
