import os
import torch
import torchaudio
import IPython.display as ipd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

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
limit = 8
labels = [name for name in os.listdir('.') if os.path.isdir(name)][:limit]
# back to default directory
os.chdir(default_dir)
print(f'Total Labels: {len(labels)} \n')
print(f'Label Names: {labels}')

def load_audio_files(path: str, label:str):
    dataset = []
    walker = sorted(str(p) for p in Path(path).glob(f'*.ogg'))

    for i, file_path in enumerate(walker):
        path, filename = os.path.split(file_path)
        speaker, _ = os.path.splitext(filename)
    
        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)
        
        target_num_samples = sample_rate * 5
        length_waveform = waveform.shape[1]

        if length_waveform > target_num_samples:
            waveform = waveform[:, sample_rate * 2:target_num_samples]
        else:
            num_missing_samples = target_num_samples - length_waveform
            last_dim_padding = (0, num_missing_samples)
            waveform = torch.nn.functional.pad(waveform, last_dim_padding)

        dataset.append([waveform, sample_rate, label, speaker])
        
    return dataset

def create_spectrogram_images(trainloader, label_dir):
    #make directory
    directory = f'./{folder}/spectrograms/{label_dir}/'
    if(os.path.isdir(directory)):
        print("Data exists for", label_dir)
    else:
        os.makedirs(directory, mode=0o777, exist_ok=True)
        
        for i, data in enumerate(trainloader):

            waveform = data[0]
            sample_rate = data[1][0]
            label = data[2]
            ID = data[3]
            
            # create transformed waveforms
            spectrogram_tensor = torchaudio.transforms.Spectrogram()(waveform)
            fig = plt.figure()
            plt.imsave(f'./{folder}/spectrograms/{label_dir}/spec_img{i}.png', spectrogram_tensor[0].log2()[0,:,:].numpy(), cmap='viridis')
            plt.close(fig)

for i, name in enumerate(labels):
    trainset = load_audio_files(f'./{folder}/train_audio/{name}', name)
    print(f'Length of #{i} {name} dataset: {len(trainset)}')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
    create_spectrogram_images(trainloader, name)
