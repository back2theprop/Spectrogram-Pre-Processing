import librosa
import numpy
import skimage.io
import os

'''
    Normalizing spectrograms -> need to find global min and max or global average min and max
'''

glbl_min = 0;
glbl_max = 0;

def calc_global_norm_stats(dataset_path, hop_length, n_mels):
    files = os.listdir(dataset_path)
    min = 0;
    max = 0;
    for file in files:
        y, sr = librosa.load(os.path.join(dataset_path, file))

        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                              n_fft=hop_length * 2, hop_length=hop_length)
        mels = numpy.log(mels + 1e-9)  # add small number to avoid log(0)

        max += mels.max()
        min += mels.min()

    min = min / len(files)
    max = max / len(files)

    return min, max

def scale_minmax(X,  mi, ma, min=0.0, max=1.0):
    X_std = (X - mi) / (ma - mi)
    X_scaled = X_std * (max - min) + min
    return X_scaled

def inverse_scale_minmax(X, mi, ma, min=0.0, max=1.0):
    X = (X - min) / (max + min)
    X_std = (X ) * (ma - mi) + mi
    return X_std

def spectrogram_image(y, sr, hop_length, n_mels, mi, ma):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                          n_fft=hop_length * 2, hop_length=hop_length)
    mels = numpy.log(mels + 1e-9)  # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, mi, ma, 0, 255).astype(numpy.uint8)
    img = numpy.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = 255 - img  # invert. make black==more energy
    # save as PNG
    return img


def inverse_spectrogram_image(img, hop_length, mi, ma):
    img = 255 - img
    img = numpy.flip(img, axis=0)
    mels = inverse_scale_minmax(img.astype(numpy.uint8), mi, ma, 0, 255)
    mels = numpy.exp(mels) - 1e-9
    audio_signal = librosa.feature.inverse.mel_to_audio(
        mels, sr=sr, n_fft=hop_length * 2, hop_length=hop_length)

    return audio_signal


if __name__ == '__main__':
    hop_length = 256  # number of samples per time-step in spectrogram
    n_mels = 128  # number of bins in spectrogram. Height of image
    time_steps = 512  # number of time-steps. Width of image

    mi, ma = calc_global_norm_stats('dataset', hop_length, n_mels)
    print(mi, ma)

    # load audio. Using example from librosa
    path = 'snare.wav'
    y, sr = librosa.load(path, sr=22050)
    if False:
        pad_length = 22050 - len(y)
        y = numpy.pad(y, (0, pad_length), 'constant', constant_values=(0, 0))
    print(len(y))
    img = spectrogram_image(y[:22050], sr=sr, hop_length=hop_length, n_mels=n_mels, mi= mi, ma= ma)
    print('here')
    skimage.io.imsave('out.png', img)

    img = skimage.io.imread('out.png')
    print('here')
    audio_signal = inverse_spectrogram_image(img, hop_length, mi, ma)
    print('here')

    import soundfile as sf

    sf.write('test.wav', audio_signal, sr)


