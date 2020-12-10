import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from pydub import AudioSegment

sound = AudioSegment.from_wav("speech.wav")
sound = sound.set_channels(1)
sound.export("speech2.wav", format="wav")
sample_rate, samples = wavfile.read('speech2.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, np.log(spectrogram), shading='auto')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()