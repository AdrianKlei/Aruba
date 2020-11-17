import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import heapq
from scipy import fftpack
import yaml


#%pylab inline --no-import-all

# Note : the first method gives back wrong amplitudes at the inverse transformation,
# hanning function could fix that, but the focus goes to the second fft method,
# which returns correct values

if __name__ == '__main__':
    # Reading in occupancy-grid informations from yaml-file
    with open("/home/adrian/Desktop/occupancy_grid_specifications.yaml") as f:
        specifications = yaml.load(f, Loader=yaml.FullLoader)
        bin_size_in_seconds = specifications['bin_size_in_seconds']
        overall_time_in_seconds = specifications['overall_time_in_seconds']


    requested_room_week = np.loadtxt("/media/adrian/Volume/occupancy_grid/cell_row_23_col_25/binary_states_array.txt")
    times_array = np.linspace(bin_size_in_seconds, overall_time_in_seconds, num=int(overall_time_in_seconds / bin_size_in_seconds), endpoint=True)    # Adjust sample time
    #requested_room_week = np.loadtxt("/home/adrian/outside_week.txt")   # default path : "/home/adrian/outside_week.txt"
    #times_array = np.linspace(60, 604800.0, num=10080, endpoint=True)
    # Adro: the following snippet is inserted to test the influences of stretching the data_array
    #requested_room = np.loadtxt("/media/adrian/Volume/occupancy_grid/cell_row_23_col_25/binary_states_array.txt")

    #times_array = np.linspace(60 * 1, 604800, num=10080, endpoint=True)

    #requested_room_week = np.array(np.arange(10080))

    # Test array is a stretched version of cell_row_23_col_25
    #for i in range(len(requested_room)):
     #   requested_room_week[i * 5:(i + 1) * 5] = requested_room[i]
        ############# end of snippet #################################################################
    plt.plot(times_array, requested_room_week)
    plt.xlabel("Time ($s$)")
    plt.ylabel("Probability ($Percent$)")
    plt.title("Week occupancy for most occupied grid cell")
    plt.show()

    print("Number of 1's in second_bedroom_week", np.count_nonzero(requested_room_week == 1))
    print("Number of 0's in second_bedroom_week", np.count_nonzero(requested_room_week == 0))

    Y = np.fft.fft(requested_room_week)
    plt.plot(Y)
    plt.title("FFT with numpy package")
    plt.show()

    N = len(Y) /2+1 # research, what exactly this line is doing
    print(Y[N-4:N+3])

    plt.plot(np.abs(Y[:N]))
    plt.show()

    dt = times_array[1] - times_array[0]
    fa = 1.0/dt
    print('dt=%.5fs (Sample time)' % dt)
    print('fa=%.5fHz (Frequency)' % fa)

    X = np.linspace(0, fa/2, N, endpoint=True) # gives us the array of frequencies according to the nyquist-criteria
    plt.plot(X, 2.0*np.abs(Y[:N])/N)
    plt.xlabel('Frequency ($Hz$)')
    plt.ylabel('Amplitude ($Unit$)')
    plt.show()
    #print(np.abs(Y[:N]))

    # Finding the N highest amplitudes of the Y[:N]-Vector and the corresponding frequencies
    test_array = np.abs(Y[:N])
    top_N_idx = heapq.nlargest(80, xrange(len(test_array)), test_array.__getitem__)

    print("The indices of the top 20 frequencies : ")
    for idx in top_N_idx:
        print(idx)

    print("---------------------------------")
    top_N_frequencies = X[top_N_idx]

    print("The top frequencies with the corresponding period times and amplitudes:")
    for idx in top_N_idx:
        print("Period time : " + str(1 / (X[idx])) +  " || Amplitude : " + str(test_array[idx]))

    print("---------------------------------")
    for i in range(5):
        print(X[i])

    model_order = 6
    threshold_amplitude = test_array[top_N_idx[model_order - 1]]

    Y[np.abs(Y) < threshold_amplitude] = 0
    filtered_sig1 = np.fft.ifft(Y)
    plt.plot(times_array, filtered_sig1)
    plt.title("Inverse Fourier Transform of order : " + str(model_order))
    plt.show()

    # now lets do all with the hanning function
    #hann = np.hanning(len(times_array))
    #plt.plot(times_array, hann * master_bedroom_one_week)
    #plt.xlabel('Time ($s$)')
    #plt.ylabel('Amplitude ($Unit$)')
    #plt.title('Signal with Hanning Window function applied')

    #Yhann = np.fft.fft(hann * master_bedroom_one_week)

    #plt.figure(figsize=(7, 3))
    #plt.subplot(121)
    #plt.plot(times_array, master_bedroom_one_week)
    #plt.title('Time Domain Signal')
    #plt.ylim(np.min(master_bedroom_one_week) * 3, np.max(master_bedroom_one_week) * 3)
    #plt.xlabel('Time ($s$)')
    #plt.ylabel('Amplitude ($Unit$)')

    #plt.subplot(122)
    #plt.plot(X, 2.0 * np.abs(Yhann[:N]) / N)
    #plt.title('Frequency Domain Signal')
    #plt.xlabel('Frequency ($Hz$)')
    #plt.ylabel('Amplitude ($Unit$)')

    #plt.annotate("FFT",
    #            xy=(0.0, 0.1), xycoords='axes fraction',
    #           xytext=(-0.8, 0.2), textcoords='axes fraction',
    #          size=30, va="center", ha="center",
    #         arrowprops=dict(arrowstyle="simple",
    #                             connectionstyle="arc3,rad=0.2"))
    #plt.tight_layout()
    #plt.show()

    #test_array = np.abs(Yhann[:N])
    #top_20_idx = heapq.nlargest(20, xrange(len(test_array)), test_array.__getitem__)

    #for idx in top_20_idx:
    #    print(idx)

    #print("---------------------------------")
    #top_20_frequencies = X[top_20_idx]

    #print("The top frequencies from Yhann with the corresponding period times :")
    #for frequency in top_20_frequencies:
    #    print(1 / frequency)

    #print("---------------------------------")
    #for i in range(5):
    #    print(X[i])

    # now we remove all frequencies except the top N from our signal and perform the inverse fourier transformation, and see if the result matches the fremenserver
    # before, for demonstration purposes, we perform the FFT with the scipy-package-method
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
    N = 6
    max_amplitudes_idx = heapq.nlargest(N, xrange(len(power[pos_mask])), power[pos_mask].__getitem__)

    print("The indices of the N highest frequencies are listed below : ")
    for i in range(len(max_amplitudes_idx)):
        print max_amplitudes_idx[i]


    # remove all frequencies that have a lower peak than the Nth highest peak

    threshold_amplitude = np.abs(sig_fft[max_amplitudes_idx[-1]])
    print("threshold amplitude is : ", threshold_amplitude)
    threshold_fft = sig_fft.copy()
    threshold_fft[np.abs(sig_fft) < threshold_amplitude] = 0
    filtered_sig2 = fftpack.ifft(threshold_fft)
    plt.figure(figsize=(6, 5))
    plt.subplot(211)
    plt.plot(times_array, filtered_sig2, label='scypi-fft')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Inverse Fourier-Transform of order 80')
    plt.subplot(212)
    plt.plot(times_array, filtered_sig1, label='numpy-fft')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.show()

    # Plot of scipy-fft for getting picture for speech 2020/10/28
    plt.figure()
    plt.plot(times_array, filtered_sig2, label='scipy-fft')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Inverse Fourier-Transform of order 80')
    plt.show()

    # test run without changing anything, just fft and then direct ifft

    test_fft = np.fft.fft(requested_room_week)
    filtered_sig3 = np.fft.ifft(test_fft)
    plt.figure(figsize=(6, 5))
    plt.plot(times_array, filtered_sig3, label='direct inverse fft')
    plt.title('Direct inverse fft')
    plt.show()

    # display filtered_sig2 as 1's and 0's
    filtered_sig2[filtered_sig2 < 0.5] = 0
    filtered_sig2[filtered_sig2 >= 0.5] = 1
    plt.figure(4)
    plt.subplot(211)
    plt.plot(times_array, requested_room_week)
    plt.title("Original binary states cell array")
    plt.subplot(212)
    plt.plot(times_array, filtered_sig2)
    plt.title("Prediction of fremen model")
    plt.show()

    # compute the error
    error = np.sum(np.abs(requested_room_week - filtered_sig2)) / len(filtered_sig2)
    print("The error sums up to : ", error)