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
os.makedirs(csv_dir,exist_ok=True)
audio_extensions = ('.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma')

all_audio_files = [f for f in os.listdir(source_dir) 
                  if os.path.isfile(os.path.join(source_dir, f)) and
                  f.lower().endswith(audio_extensions)]

random.seed(10)
random_audio_files = random.sample(all_audio_files, 10)

for file_name in random_audio_files:
    full_file_name = os.path.join(source_dir, file_name)
    shutil.copy(full_file_name, dest_dir)

os.makedirs(plots_dir, exist_ok=True)


audio_files = [f for f in os.listdir(dest_dir) if os.path.isfile(os.path.join(dest_dir, f))]


def extract_features(y, sr, file_name):
    
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)  
    mfcc_dict = {f'MFCC_{i+1}': value for i, value in enumerate(mfcc)}

    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    chroma_dict = {f'chroma_{i+1}': value for i, value in enumerate(chroma)}

    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    spectral_contrast_dict = {f'spectral_contrast_{i+1}': value for i, value in enumerate(spectral_contrast)}
    
    features = {
        'file_name': file_name,
        **mfcc_dict,
        **chroma_dict,
        **spectral_contrast_dict,
        'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
        'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    }
    
    return features

features_list = []

def save_waveform(y, sr, file_name):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform of {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plot_path = os.path.join(plots_dir, f"{file_name}_waveform.png")
    plt.savefig(plot_path)
    plt.close()

def save_spectrogram(y, sr, file_name):
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram of {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plot_path = os.path.join(plots_dir, f"{file_name}_spectrogram.png")
    plt.savefig(plot_path)
    plt.close()


def save_mfcc(y, sr, file_name):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title(f"MFCC of {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("MFCC Coefficients")
    plot_path = os.path.join(plots_dir, f"{file_name}_mfcc.png")
    plt.savefig(plot_path)
    plt.close()

def save_chroma(y, sr, file_name):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
    plt.colorbar()
    plt.title(f"Chroma Features of {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Chroma")
    plot_path = os.path.join(plots_dir, f"{file_name}_chroma.png")
    plt.savefig(plot_path)
    plt.close()

def save_spectral_contrast(y, sr, file_name):
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectral_contrast, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title(f"Spectral Contrast of {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Spectral Contrast")
    plot_path = os.path.join(plots_dir, f"{file_name}_spectral_contrast.png")
    plt.savefig(plot_path)
    plt.close()

def save_zero_crossing_rate(y, sr, file_name):
    zero_crossings = librosa.feature.zero_crossing_rate(y)
    plt.figure(figsize=(10, 4))
    plt.plot(zero_crossings[0])
    plt.title(f"Zero-Crossing Rate of {file_name}")
    plt.xlabel("Frames")
    plt.ylabel("Zero-Crossing Rate")
    plot_path = os.path.join(plots_dir, f"{file_name}_zero_crossing_rate.png")
    plt.savefig(plot_path)
    plt.close()

def save_spectral_rolloff(y, sr, file_name):
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
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
    y, sr = librosa.load(file_path, sr=None) 

    save_waveform(y, sr, f_name)
    save_spectrogram(y, sr, f_name)
    save_mfcc(y, sr, f_name)
    save_chroma(y, sr, f_name)
    save_spectral_contrast(y, sr, f_name)
    save_zero_crossing_rate(y, sr, f_name)
    save_spectral_rolloff(y, sr, f_name)
    features = extract_features(y, sr, f_name)
    features_list.append(features)

features_df = pd.DataFrame(features_list)
# print(features_list)
csv_path = os.path.join(csv_dir, 'audio_features.csv')
if not os.path.exists(csv_path):
    features_df.to_csv(csv_path, index=False)
