import hashlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from operator import itemgetter
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
    ampls = spec[detected_peaks]
    ampls = ampls.flatten()
    freqs, times = np.where(detected_peaks)
    filter_idxs = np.where(ampls > 10)

    freqs_filter = freqs[filter_idxs]
    times_filter = times[filter_idxs]

    return list(zip(freqs_filter, times_filter))

def generate_hashes(peaks, fan_value):
    hashes = []
    peaks.sort(key=itemgetter(1))
    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if (i + j) < len(peaks):
                freq1 = peaks[i][0]
                freq2 = peaks[i + j][0]
                time1 = peaks[i][1]
                time2 = peaks[i + j][1]
                time_delta = time2 - time1
                if 0 <= time_delta <= 200:
                    h = hashlib.sha1(f"{str(freq1)}|{str(freq2)}|{str(time_delta)}".encode('utf-8'))
                    hashes.append(h.hexdigest()[0:20])
                    # hashes.append(h.hexdigest()[0:20], time1) - с помощью time1 можно подвинуть сдвинутый фригмент, но в примитивном виде он тлько мешает
    return hashes

def get_sound(name):
    sound = AudioSegment.from_wav(name + ".wav")
    sound = sound.set_channels(1)
    sound.export(name + "_mono.wav", format="wav")
    sample_rate, samples = wavfile.read(name + "_mono.wav")
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
    fig, ax = plt.subplots()
    ax.imshow(spectrogram, interpolation='nearest', aspect='auto')
    ax.scatter(times2, freqs, color='red')
    plt.ylabel('Частоты')
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
    biggest_const = 0
    constel = 0
    print(len(hash1))
    for i in range(0, len(hash1)):
        for j in range(0, len(hash2)):
            if hash1[i] == hash2[j]:
                print(i)
                print(j)
                print("***************************")
                if i > 1 and j > 1 and hash1[i-1] == hash2[j-1]:
                    constel = constel + 1
                    if constel > biggest_const:
                        biggest_const = constel
                if i > 1 and j > 1 and hash1[i-1] != hash2[j-1]:
                    constel = 0
                equal = equal + 1
    print(equal)
    print(biggest_const)
    return

def find_constellations(hash1, hash2):
    print(len(hash1))
    biggest_const = 0
    constel = 0
    for i in range(0, len(hash1)):
        for j in range(0, len(hash2)):
            if hash1[i] == hash2[j]:
                k = i + 1
                p = j + 1
                while k < len(hash1) and p < len(hash2) and hash1[k] == hash2[p]:
                    constel = constel + 1
                    k = k + 1
                    p = p + 1
                if constel > biggest_const:
                    biggest_const = constel
                constel = 0

    print(biggest_const)


find_constellations(get_sound("female"), get_sound("mix3"))
#get_sound("mix")


