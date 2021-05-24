import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import librosa
from librosa import display

y, sr = librosa.load('Kick.wav')
print(sr)
print(len(y))
S = librosa.feature.melspectrogram(y=y[:22050], sr=sr, S=None, n_fft=2048, hop_length=173, win_length=1024)

print(S.shape)
S_dB = librosa.power_to_db(S, ref=np.max)

fig, ax = plt.subplots()

S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr,
                         fmax=8000, ax=ax)
print('img',img.shape)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')

plt.show()

S = np.load('out.npz.npy')
print(S_dB.shape)

S = librosa.feature.inverse.mel_to_stft(S)
y_inv = librosa.griffinlim(S)
fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
librosa.display.waveplot(y, sr=sr, color='b', ax=ax[0])
ax[0].set(title='Original', xlabel=None)
ax[0].label_outer()
librosa.display.waveplot(y_inv, sr=sr, color='g', ax=ax[1])
ax[1].set(title='Griffin-Lim reconstruction', xlabel=None)
ax[1].label_outer()
plt.show()
