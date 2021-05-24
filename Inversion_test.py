import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import librosa
from librosa import display
import soundfile as sf

samples, sample_rate = librosa.load('Kick2.wav')

melspectrogram = librosa.feature.melspectrogram(
        y=samples[:22050], sr=sample_rate,  n_fft=2048, hop_length=173)

print('melspectrogram.shape', melspectrogram.shape)
S_dB = librosa.power_to_db(melspectrogram, ref=np.max)
print(S_dB.shape)
audio_signal = librosa.feature.inverse.mel_to_audio(
    melspectrogram, sr=sample_rate, n_fft=2048, hop_length=173)
print(audio_signal.shape)

sf.write('test.wav', audio_signal, sample_rate)

import png
png.from_array(S_dB, 'L').save('spec.png')