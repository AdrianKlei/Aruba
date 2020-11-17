import numpy as np
import pymongo
import yaml
import numpy as np
from datetime import datetime
import time
from matplotlib import pyplot as plt
from scipy import fftpack
import pymongo
from tqdm import tqdm
import heapq
import math

client = pymongo.MongoClient('localhost', 27017)
print('Connection to database has been established.')
db = client['uol_database_float']
collection = db['cell_recordings']

data = collection.find({"name":"cell_row_23_col_25"})
print(data[0])
print(data[1])

states = [entry["state"] for entry in data]
print(len(states))
weeks = np.array_split(states, 3)
print(len(weeks[0]))

plt.figure()
plt.subplot(311)
plt.plot(weeks[0])
plt.title("First week data")
plt.ylim(-0.2, 100)
plt.subplot(312)
plt.plot(weeks[1])
plt.title("Second week data")
plt.ylim(-0.2, 100)
plt.subplot(313)
plt.plot(weeks[2])
plt.title("Third week")
plt.ylim(-0.2, 100)
plt.show()

# Perform a FFT on one example week, and then test the accuracy on itself
requested_room_week = weeks[0]
times_array = np.linspace(600, 604800, num= int(604800/600))
errors_array = []

for i in range(100):
    sig_fft = fftpack.fft(requested_room_week) # in contrast to the example, here is used the hann window-function
    power = np.abs(sig_fft)**2
    sample_freq = fftpack.fftfreq(requested_room_week.size, d=times_array[1] - times_array[0])

    #plt.figure(figsize=(6, 5))
    #plt.plot(sample_freq, power)
    #plt.xlabel('Frequency [Hz]')
    #plt.ylabel('plower')

    pos_mask = np.where(sample_freq >= 0) # 0 needs to be included, as it represents the DC part of our signal
    freqs = sample_freq[pos_mask]
    peak_freq = freqs[power[pos_mask].argmax()]
    print("The peak corresponds to the frequency : ", peak_freq)
    plt.show()

    #identificate the N frequencies with the highest amplitude
    N = i+1
    max_amplitudes_idx = heapq.nlargest(N, xrange(len(power[pos_mask])), power[pos_mask].__getitem__)

    print("The indices of the N highest frequencies are listed below : ")
    for i in range(len(max_amplitudes_idx)):
        print max_amplitudes_idx[i]


    # remove all frequencies that have a lower peak than the Nth highest peak

    threshold_amplitude = np.abs(sig_fft[max_amplitudes_idx[-1]])
    print("threshold amplitude is : ", threshold_amplitude)
    threshold_fft = sig_fft.copy()
    threshold_fft[np.abs(sig_fft) < threshold_amplitude] = 0
    filtered_sig = fftpack.ifft(threshold_fft)

    #plt.figure(figsize=(6, 10))
    #plt.subplot(211)

    #plt.plot(times_array, weeks[0])
    #plt.ylim(-0.2, 100)
    #plt.subplot(212)

    #plt.plot(times_array, filtered_sig, label='scypi-fft')
    #plt.ylim(-0.2, 100)
    #plt.xlabel('Time [s]')
    #plt.ylabel('Amplitude')
    #plt.title('Inverse Fourier-Transform of order 80')

    #plt.show()

    # Compute the root mean squared error of the predictions
    original_data = weeks[0]
    predictions = filtered_sig

    rmse = math.sqrt(np.sum((original_data - predictions)**2)/len(original_data))
    #rmse = np.sqrt(np.sum(np.square(original_data - predictions)) / len(original_data))

    errors_array.append(rmse)
    print(rmse)

plt.figure()
plt.plot(errors_array)
plt.title("Errors over model numbers")
plt.show()

# Now we compute the IFFT for ONE specific order to compare the results with them from the fremenserver
order = 11
sig_fft = fftpack.fft(requested_room_week) # in contrast to the example, here is used the hann window-function
power = np.abs(sig_fft)**2
sample_freq = fftpack.fftfreq(requested_room_week.size, d=times_array[1] - times_array[0])

plt.figure(figsize=(6, 5))
plt.plot(sample_freq, power)
plt.xlabel('Frequency [Hz]')
plt.ylabel('plower')

pos_mask = np.where(sample_freq >= 0) # 0 needs to be included, as it represents the DC part of our signal
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]
print("The peak corresponds to the frequency : ", peak_freq)
plt.show()

#identificate the N frequencies with the highest amplitude
N = order
max_amplitudes_idx = heapq.nlargest(N, xrange(len(power[pos_mask])), power[pos_mask].__getitem__)

print("The indices of the N highest frequencies are listed below : ")
for i in range(len(max_amplitudes_idx)):
    print max_amplitudes_idx[i]


# remove all frequencies that have a lower peak than the Nth highest peak

threshold_amplitude = np.abs(sig_fft[max_amplitudes_idx[-1]])
print("threshold amplitude is : ", threshold_amplitude)
threshold_fft = sig_fft.copy()
threshold_fft[np.abs(sig_fft) < threshold_amplitude] = 0
filtered_sig = fftpack.ifft(threshold_fft)

plt.figure(figsize=(6, 10))
plt.subplot(211)

plt.plot(times_array, weeks[0])
plt.ylim(-0.2, 100)
plt.subplot(212)

plt.plot(times_array, filtered_sig, label='scypi-fft')
plt.ylim(-0.2, 100)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Inverse Fourier-Transform of order N')

plt.show()

print("Errors array : ", errors_array[:20])