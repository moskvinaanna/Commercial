import hashlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from typing import List, Tuple
from scipy import signal
from scipy.io import wavfile
from pydub import AudioSegment
from scipy.signal import find_peaks
from scipy.misc import electrocardiogram
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (binary_erosion,
                                      generate_binary_structure,
                                      iterate_structure)


def get_2D_peaks(spec):
    struct = generate_binary_structure(2, 2)
    neighborhood = iterate_structure(struct, 10)
    local_max = maximum_filter(spec, footprint=neighborhood) == spec
    background = (spec == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max != eroded_background
    amps = spec[detected_peaks]
    freqs, times = np.where(detected_peaks)
    amps = amps.flatten()
    filter_idxs = np.where(amps > 10)

    freqs_filter = freqs[filter_idxs]
    times_filter = times[filter_idxs]

    return list(zip(freqs_filter, times_filter))

def generate_hashes(peaks, fan_value):
    idx_freq = 0
    idx_time = 1
    hashes = []
    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if (i + j) < len(peaks):

                freq1 = peaks[i][idx_freq]
                freq2 = peaks[i + j][idx_freq]
                t1 = peaks[i][idx_time]
                t2 = peaks[i + j][idx_time]
                t_delta = t2 - t1

                if 0 <= t_delta <= 200:
                    h = hashlib.sha1(f"{str(freq1)}|{str(freq2)}|{str(t_delta)}".encode('utf-8'))

                    hashes.append((h.hexdigest()[0:2], t1))

    return hashes

def get_sound(name):
    sound = AudioSegment.from_wav(name + ".wav")
    sound = sound.set_channels(1)
    sound.export(name + "_mono.wav", format="wav")
    sample_rate, samples = wavfile.read(name + "_mono.wav")
    frequencies, times, spectrogram2 = signal.spectrogram(samples, sample_rate)
    spectrogram = mlab.specgram(
        samples,
        NFFT= 4096,
        Fs=44100,
        window=mlab.window_hanning,
        noverlap=int(4096*0.5))[0]
    spectrogram = 10*np.log10(spectrogram, out=np.zeros_like(spectrogram), where=(spectrogram != 0))
    freqs = []
    times2 = []
    peaks = get_2D_peaks(spectrogram)
    print(peaks)
    hash = generate_hashes(peaks, 10)
    print("***************************")
    print(hash)
    for i in range(0, len(peaks)):
        freqs.append(peaks[i][0])
        times2.append(peaks[i][1])

    #plt.pcolormesh(times, frequencies, np.log(spectrogram), shading='auto')
    #plt.plot(spectrogram)

    # data_1D = samples.flatten()

    fig, ax = plt.subplots()
    ax.imshow(spectrogram, interpolation='nearest', aspect='auto')
    ax.scatter(times2, freqs, color='red')
    #pxx,  freq, t, cax = ax.specgram(spectrogram,  NFFT= 4096, Fs=44100, window=mlab.window_hanning, noverlap=int(4096 * 0.5))
    # fig.colorbar(cax)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Единицы времени')
    plt.gca().invert_yaxis()
    plt.show()

    return hash

def compare_hashes2(hash1, hash2):
    if (hash1 == hash2):
        print("equal")
    else:
        print("different")
    return
def compare_hashes(hash1, hash2):
    equal = 0
    print(len(hash1))
    for i in range(0, len(hash1)):
        for j in range(0, len(hash2)):
            if hash1[i] == hash2[j]:
                equal = equal + 1
    print(equal)
    return

compare_hashes(get_sound("male"), get_sound("mix3"))
#get_sound("mix")


