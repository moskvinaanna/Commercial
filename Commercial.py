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


def get_2D_peaks(arr2D: np.array, plot: bool = False, amp_min: int = 10) \
        -> List[Tuple[List[int], List[int]]]:
    """
    Extract maximum peaks from the spectogram matrix (arr2D).
    :param arr2D: matrix representing the spectogram.
    :param plot: for plotting the results.
    :param amp_min: minimum amplitude in spectrogram in order to be considered a peak.
    :return: a list composed by a list of frequencies and times.
    """
    # Original code from the repo is using a morphology mask that does not consider diagonal elements
    # as neighbors (basically a diamond figure) and then applies a dilation over it, so what I'm proposing
    # is to change from the current diamond figure to a just a normal square one:
    #       F   T   F           T   T   T
    #       T   T   T   ==>     T   T   T
    #       F   T   F           T   T   T
    # In my local tests time performance of the square mask was ~3 times faster
    # respect to the diamond one, without hurting accuracy of the predictions.
    # I've made now the mask shape configurable in order to allow both ways of find maximum peaks.
    # That being said, we generate the mask by using the following function
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html
    struct = generate_binary_structure(2, 2)

    #  And then we apply dilation using the following function
    #  http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.iterate_structure.html
    #  Take into account that if PEAK_NEIGHBORHOOD_SIZE is 2 you can avoid the use of the scipy functions and just
    #  change it by the following code:
    #  neighborhood = np.ones((PEAK_NEIGHBORHOOD_SIZE * 2 + 1, PEAK_NEIGHBORHOOD_SIZE * 2 + 1), dtype=bool)
    neighborhood = iterate_structure(struct, 10)

    # find local maxima using our filter mask
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D

    # Applying erosion, the dejavu documentation does not talk about this step.
    background = (arr2D == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # Boolean mask of arr2D with True at peaks (applying XOR on both matrices).
    detected_peaks = local_max != eroded_background

    # extract peaks
    amps = arr2D[detected_peaks]
    freqs, times = np.where(detected_peaks)

    # filter peaks
    amps = amps.flatten()

    # get indices for frequency and time
    filter_idxs = np.where(amps > amp_min)

    freqs_filter = freqs[filter_idxs]
    times_filter = times[filter_idxs]

    return list(zip(freqs_filter, times_filter))

sound = AudioSegment.from_wav("audio.wav")
sound = sound.set_channels(1)
sound.export("audio2.wav", format="wav")
sample_rate, samples = wavfile.read('audio2.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
freqs = []
times2 = []
peaks = get_2D_peaks(spectrogram, plot=False)
print(peaks)
for i in range(0, len(peaks)):
    freqs.append(peaks[i][0])
    times2.append(peaks[i][1])


#plt.pcolormesh(times, frequencies, np.log(spectrogram), shading='auto')
plt.plot(spectrogram)
plt.scatter(times2, freqs)
#data_1D = samples.flatten()
for i in range(0, 32):
    spectrogram[1][i] = 0
fig, ax = plt.subplots()
#pxx,  freq, t, cax = ax.specgram(spectrogram[1], NFFT = 64, Fs = 64, noverlap=32)
#fig.colorbar(cax)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.gca().invert_yaxis()
plt.show()