import math
import os
import torch
import torchaudio
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.transforms import Bbox


matplotlib.use('Agg')

default_dir = os.getcwd()
folder = 'birdclef-2022'
print(f'Data directory will be: {default_dir}/{folder}')

if os.path.isdir(folder):
    print("Data folder exists.")
else:
    print("Creating folder.")
    os.mkdir(folder) 

os.chdir(f'./{folder}/train_audio/')

# Sample limit
limit = 20
min_sample = 20
duration = 0 # set 0 for no trimming
labels = [name for name in os.listdir('.') if os.path.isdir(name)]

done = 0

# back to default directory
os.chdir(default_dir)

def mix_down_if_necessary(signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal

def load_audio_files(path: str, label:str):
    global done

    directory = f'./{folder}/spectrograms/{label}/'
    walker = sorted(str(p) for p in Path(path).glob(f'*.ogg'))

    if len(walker) >= min_sample:
        print("Gen:", label)
        if os.path.isdir(directory):
            done += 1
            return
        os.makedirs(directory, mode=0o777, exist_ok=True)

        for i, file_path in enumerate(walker):
            path, filename = os.path.split(file_path)
            print(filename, " is created")
            # Load audio
            waveform, sample_rate = torchaudio.load(file_path)
            waveform = mix_down_if_necessary(waveform)

            length_waveform = waveform.shape[1]

            if duration:
                target_num_samples = math.ceil(sample_rate * duration)
                skip = sample_rate * 2

                if length_waveform > target_num_samples + skip:
                    waveform = waveform[:, skip:target_num_samples]
                elif length_waveform > target_num_samples:
                    waveform = waveform[:, :target_num_samples]
                else:
                    num_missing_samples = target_num_samples - length_waveform - (sample_rate * 2)
                    last_dim_padding = (0, num_missing_samples)
                    waveform = torch.nn.functional.pad(waveform, last_dim_padding)

            # create transformed waveforms
            spectrogram_tensor = torchaudio.transforms.Spectrogram()(waveform)
            # plt.show()
            # plt.imsave(f'./{folder}/spectrograms/{label}/spec_img{i}.png', spectrogram_tensor.log2()[0,:,:].numpy(), cmap='viridis')

            my_dpi = 101
            h = 201
            w = 160 * (duration if duration else length_waveform / sample_rate)
            fig, ax = plt.subplots(1, figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
            ax.set_position([0, 0, 1, 1])
            ax.imshow(spectrogram_tensor.log2()[0,:,:].numpy())
            ax.axis("off")
            plt.savefig(f'./{folder}/spectrograms/{label}/spec_img{i}.png', transparent = True, bbox_inches=Bbox([[0, 0], [w/my_dpi, h/my_dpi]]),
                dpi=my_dpi)
            plt.close(fig)

        done += 1

for i, name in enumerate(labels):
    if (done < limit):
        load_audio_files(f'./{folder}/train_audio/{name}', name)   
    else:
        break
# load_audio_files(f'./{folder}/train_audio/barpet', 'barpet')
