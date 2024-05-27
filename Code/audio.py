import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import shutil

source_dir = 'Input_audio'
dest_dir = 'Selected_audio_file'
plots_dir = 'audio_plots'
csv_dir = 'csv_file'

os.makedirs(dest_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)
audio_extensions = ('.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma')

all_audio_files = [f for f in os.listdir(source_dir) 
                  if os.path.isfile(os.path.join(source_dir, f)) and
                  f.lower().endswith(audio_extensions)]

random.seed(50)
random_audio_files = random.sample(all_audio_files, 10)

def clear_directory(directory):
    if os.path.exists(directory) and os.path.isdir(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                os.remove(item_path)


clear_directory(dest_dir)
clear_directory(plots_dir)

for file_name in random_audio_files:
    full_file_name = os.path.join(source_dir, file_name)
    shutil.copy(full_file_name, dest_dir)

os.makedirs(plots_dir, exist_ok=True)

audio_files = []

for item in os.listdir(dest_dir):
    if os.path.isfile(os.path.join(dest_dir, item)):
        audio_files.append(item)

def extract_features(audio_data, sampling_rate, file_name):
    mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=15), axis=1)  
    mfcc_dict = {}
    for i, value in enumerate(mfcc):
        mfcc_dict[f'MFCC_{i+1}'] = value

    chroma = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=sampling_rate), axis=1)
    chroma_dict = {}
    for i, value in enumerate(chroma):
        chroma_dict[f'chroma_{i+1}'] = value

    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio_data, sr=sampling_rate), axis=1)
    spectral_contrast_dict = {}
    for i, value in enumerate(spectral_contrast):
        spectral_contrast_dict[f'spectral_contrast_{i+1}'] = value    
        
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sampling_rate))

    features = {
        'file_name': file_name,  
    }
    for i, value in enumerate(mfcc):
        features[f'MFCC_{i+1}'] = value

    for i, value in enumerate(chroma):
        features[f'chroma_{i+1}'] = value

    for i, value in enumerate(spectral_contrast):
        features[f'spectral_contrast_{i+1}'] = value

    features['zero_crossing_rate'] = zero_crossing_rate
    features['spectral_rolloff'] = spectral_rolloff
        
    return features

features_list = []

def save_waveform(audio_data, sampling_rate, file_name):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio_data, sr=sampling_rate)
    plt.title(f"Waveform of {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plot_path = os.path.join(plots_dir, f"{file_name}_waveform.png")
    plt.savefig(plot_path)
    plt.close()

def save_spectrogram(audio_data, sampling_rate, file_name):
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    librosa.display.specshow(D, sr=sampling_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram of {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plot_path = os.path.join(plots_dir, f"{file_name}_spectrogram.png")
    plt.savefig(plot_path)
    plt.close()

def save_mfcc(audio_data, sampling_rate, file_name):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=15)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sampling_rate, x_axis='time')
    plt.colorbar()
    plt.title(f"MFCC of {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("MFCC Coefficients")
    plot_path = os.path.join(plots_dir, f"{file_name}_mfcc.png")
    plt.savefig(plot_path)
    plt.close()

def save_chroma(audio_data, sampling_rate, file_name):
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sampling_rate)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, sr=sampling_rate, x_axis='time', y_axis='chroma')
    plt.colorbar()
    plt.title(f"Chroma Features of {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Chroma")
    plot_path = os.path.join(plots_dir, f"{file_name}_chroma.png")
    plt.savefig(plot_path)
    plt.close()

def save_spectral_contrast(audio_data, sampling_rate, file_name):
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sampling_rate)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectral_contrast, sr=sampling_rate, x_axis='time')
    plt.colorbar()
    plt.title(f"Spectral Contrast of {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Spectral Contrast")
    plot_path = os.path.join(plots_dir, f"{file_name}_spectral_contrast.png")
    plt.savefig(plot_path)
    plt.close()

def save_zero_crossing_rate(audio_data, sampling_rate, file_name):
    zero_crossings = librosa.feature.zero_crossing_rate(audio_data)
    plt.figure(figsize=(10, 4))
    plt.plot(zero_crossings[0])
    plt.title(f"Zero-Crossing Rate of {file_name}")
    plt.xlabel("Frames")
    plt.ylabel("Zero-Crossing Rate")
    plot_path = os.path.join(plots_dir, f"{file_name}_zero_crossing_rate.png")
    plt.savefig(plot_path)
    plt.close()

def save_spectral_rolloff(audio_data, sampling_rate, file_name):
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sampling_rate)
    plt.figure(figsize=(10, 4))
    plt.plot(spectral_rolloff[0])
    plt.title(f"Spectral Roll-off of {file_name}")
    plt.xlabel("Frames")
    plt.ylabel("Spectral Roll-off (Hz)")
    plot_path = os.path.join(plots_dir, f"{file_name}_spectral_rolloff.png")
    plt.savefig(plot_path)
    plt.close()

for f_name in audio_files:
    file_path = os.path.join(dest_dir, f_name)
    audio_data, sampling_rate = librosa.load(file_path)

    save_waveform(audio_data, sampling_rate, f_name)
    save_spectrogram(audio_data, sampling_rate, f_name)
    save_mfcc(audio_data, sampling_rate, f_name)
    save_chroma(audio_data, sampling_rate, f_name)
    save_spectral_contrast(audio_data, sampling_rate, f_name)
    save_zero_crossing_rate(audio_data, sampling_rate, f_name)
    save_spectral_rolloff(audio_data, sampling_rate, f_name)
    features = extract_features(audio_data, sampling_rate, f_name)
    features_list.append(features)

features_df = pd.DataFrame(features_list)
csv_path = os.path.join(csv_dir, 'audio_features.csv')

if os.path.exists(csv_path):
    os.remove(csv_path)
features_df.to_csv(csv_path, index=False)
