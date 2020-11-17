import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt
import heapq
import math


aruba_dataset = np.loadtxt("/home/adrian/Desktop/locations.min")
master_bedroom_occupancy = np.copy(aruba_dataset)
master_bedroom_occupancy[aruba_dataset == 0] = 1
master_bedroom_occupancy[aruba_dataset != 0] = 0

# Split the data into the 16 weeks
master_bedroom_weeks_list = np.array_split(master_bedroom_occupancy, 16)

master_bedroom_week_0 = master_bedroom_weeks_list[0]

plt.figure()
plt.plot(master_bedroom_week_0)
plt.title("Master bedroom week 0: occurrences every 60 seconds")
plt.show()

# We create another array, with intervals of 10 minutes, summing up the number of occurrences in the master bedroom
# within this time intervall, thus we don't have a binary array anymore

master_bedroom_week_0_intervals = np.zeros(len(master_bedroom_week_0) / 10)
times_array_week_0_intervals = np.linspace(600, 604800, num=int(604800/600))
print(len(master_bedroom_week_0_intervals))
for i in range(len(master_bedroom_week_0_intervals)):
    master_bedroom_week_0_intervals[i] = sum(master_bedroom_week_0[i*10:(i+1)*10])
plt.figure()
plt.plot(master_bedroom_week_0_intervals)
plt.title("Master bedroom week 0: accumulated occurrences every 600 seconds")
plt.show()

# We perform a FFT on the interval array, extract the N frequencies with the highest amplitudes, and perform an
# IFFT based on that N frequencies

requested_room_week = master_bedroom_week_0_intervals
sig_fft = fftpack.fft(requested_room_week)
power = np.abs(sig_fft)**2
sample_freq = fftpack.fftfreq(requested_room_week.size, d=times_array_week_0_intervals[1] - times_array_week_0_intervals[0])

pos_mask = np.where(sample_freq >= 0) # 0 needs to be included, as it represents the DC part of our signal
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]
print("The peak corresponds to the frequency : ", peak_freq)
#plt.show()

# Identificate the N frequencies with the highest amplitude
N = 6
max_amplitudes_idx = heapq.nlargest(N, xrange(len(power[pos_mask])), power[pos_mask].__getitem__)

print("The indices of the N highest frequencies are listed below : ")
for i in range(len(max_amplitudes_idx)):
    print max_amplitudes_idx[i]


# Remove all frequencies that have a lower peak than the Nth highest peak

threshold_amplitude = np.abs(sig_fft[max_amplitudes_idx[-1]])
print("threshold amplitude is : ", threshold_amplitude)
threshold_fft = sig_fft.copy()
threshold_fft[np.abs(sig_fft) < threshold_amplitude] = 0
filtered_sig = fftpack.ifft(threshold_fft)

plt.figure()
plt.plot(times_array_week_0_intervals, filtered_sig)
plt.title("Signal after IFFT of N frequencies with highest amplitude")
plt.show()

# Round the filtered signal to the nearest integer, set all negative values equal 0 (logical thinking)
filtered_sig_rounded = np.rint(filtered_sig)
filtered_sig_rounded[filtered_sig_rounded < 0] = 0

plt.figure()
plt.plot(times_array_week_0_intervals, filtered_sig_rounded)
plt.title("Rounded IFFT-Signal")
plt.show()

print(len(master_bedroom_week_0_intervals))
print(len(filtered_sig_rounded))


plt.figure()
plt.subplot(211)
plt.plot(times_array_week_0_intervals, master_bedroom_week_0_intervals)
plt.title("Original data of week 0")
plt.subplot(212)
plt.plot(times_array_week_0_intervals, filtered_sig_rounded)
plt.title("Prediction model of order N")
plt.show()

# Compute the root-mean-squared-error (RMSE) of the prediction-model
rmse = math.sqrt(np.sum((master_bedroom_week_0_intervals - filtered_sig_rounded)**2)/len(filtered_sig_rounded))
print("rmse sums up to : ", rmse)
avg_occurrences = np.sum(master_bedroom_week_0_intervals) / len(master_bedroom_week_0_intervals)
exit()
# We perform a FFT with the 'l addition amplitude model' (AAM) expansion
input_signal = master_bedroom_week_0_intervals
total = 6
S = []
k = 1
# Get the frequency zero(w1 = 0)
sig_fft = fftpack.fft(input_signal)
print(sig_fft[0])
print(sig_fft[1:3])

omega_k = sig_fft[0]
S.append([np.abs(omega_k), np.angle(omega_k), 0])
while k <= 5:
    # Get the frequency with the highest amplitude
    sig_fft = fftpack.fft(input_signal)
    omega_k = np.argmax(np.abs(sig_fft))
    # Update S with omega_k
    already_exists = False
    for i in range(len(S)):
        if S[i][2] == omega_k:
            already_exists = True
            S[i][0] = S[i][0] + np.abs(sig_fft[omega_k])
            S[i][1] = (S[i][1] + np.angle(sig_fft[omega_k]))/2
    if not already_exists:
        S.append([np.abs(sig_fft[omega_k]), np.angle(sig_fft[omega_k]), omega_k])
        k+=1
    # Create a cosine signal from omega_k and substract
    cosine_signal = np.zeros(len(input_signal))
    cosine_signal[:] = np.abs(sig_fft[omega_k])*math.cos(2*math.pi*omega_k + np.angle(sig_fft[omega_k]))
    input_signal = input_signal - cosine_signal

print("Check")
print(S)




