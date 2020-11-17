import numpy as np
import yaml
import matplotlib.pyplot as plt
import math
from scipy import fftpack
import heapq
import seaborn as sns
from tqdm import tqdm
from matplotlib.patches import Rectangle
import pandas as pd

aruba_dataset = np.loadtxt("/home/adrian/Desktop/locations.min")
master_bedroom_occupancy = np.copy(aruba_dataset)
master_bedroom_occupancy[aruba_dataset == 0] = 1
master_bedroom_occupancy[aruba_dataset != 0] = 0


# Split the data into the 16 weeks
master_bedroom_weeks_list = np.array_split(master_bedroom_occupancy, 16)

master_bedroom_week_0 = master_bedroom_weeks_list[0]

print("Length of master bedroom weeks list :", len(master_bedroom_weeks_list))
outside_weeks_list_transformed = []
times_array = np.linspace(60, 604800, num=10080)
# Now we perform the FFT (scipy-package) for all 16 week chunks and save the results to a list
for week in master_bedroom_weeks_list:
    requested_room_week = week
    sig_fft = fftpack.fft(requested_room_week)
    power = np.abs(sig_fft)**2
    sample_freq = fftpack.fftfreq(requested_room_week.size, d=times_array[1] - times_array[0])

    pos_mask = np.where(sample_freq >= 0) # 0 needs to be included, as it represents the DC part of our signal
    freqs = sample_freq[pos_mask]
    peak_freq = freqs[power[pos_mask].argmax()]
    print("The peak corresponds to the frequency : ", peak_freq)
    plt.show()

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

    filtered_sig[filtered_sig < 0.5] = 0
    filtered_sig[filtered_sig >= 0.5] = 1
    error = np.sum(np.abs(requested_room_week - filtered_sig)) / len(filtered_sig)
    print("The error sums up to : ", error)
    outside_weeks_list_transformed.append(filtered_sig)

# Now we have to perform a validation for each week with each several transformed week
# Therefore we create a matrix
validation_matrix = np.zeros((len(master_bedroom_weeks_list), len(master_bedroom_weeks_list)))
for i in tqdm(range(len(outside_weeks_list_transformed)), desc="Prediction model progress : "):
    for j in range(len(master_bedroom_weeks_list)):
        prediction_model = outside_weeks_list_transformed[i]
        active_week = master_bedroom_weeks_list[j]
        error = np.sum(np.abs(prediction_model - active_week)) / len(prediction_model)
        accuracy = 1 - error
        validation_matrix[i, j] = accuracy

# Plotting the confusion matrix and highlight the minimum error week for each prediction model.
columns_text = [str(num) for num in range(len(master_bedroom_weeks_list))]
index_text = [str(num) for num in range(len(outside_weeks_list_transformed))]
arr_data_frame = pd.DataFrame(validation_matrix, columns=columns_text, index=index_text)

fig, ax = plt.subplots(figsize=(16, 16), facecolor='w', edgecolor='k')
ax = sns.heatmap(arr_data_frame, annot=True, vmax=1.0, vmin=0, cbar=True, fmt='.4g', ax=ax)

# Highlighting the maximum predicition value in each row
row_max = arr_data_frame.idxmax(axis=1)

for row, index in enumerate(index_text):
    position = arr_data_frame.columns.get_loc(row_max[index])
    ax.add_patch(Rectangle((position, row),1,1, fill=False, edgecolor='red', lw=3))
plt.title("Accuracy of prediction models for 16 different weeks")
plt.xlabel('Week number')
plt.ylabel('Prediction model number')
#plt.savefig('kitchen_confusion_matrix_order_5.png', dpi= 500)
plt.show()

# Now we compare the predictions of the models
# We have a model trained with data of one week, and a model trained with data of 10 weeks
week_probabilities = np.loadtxt("/home/adrian/Desktop/week_probabilities.txt")
order_0_predictions = week_probabilities[:16]
order_5_predictions = week_probabilities[80:96]
order_1_predictions = week_probabilities[16:32]

plt.figure()
plt.subplot(411)
plt.plot(order_5_predictions[0])
plt.subplot(412)
plt.plot(order_5_predictions[1])
plt.subplot(413)
plt.plot(order_5_predictions[2])
plt.subplot(414)
plt.plot(np.array(order_5_predictions[0] - np.array(order_5_predictions[1])))
plt.show()
print("---------------------------")
print(np.array_equal(np.array(order_5_predictions[0]), np.array(order_5_predictions[1])))
# Compare master_bedroom_weeks_list and errors_order_x (both shape (16, 10080))

accuracies_order_0 = []
accuracies_order_5 = []
accuracies_order_1 = []

for i in range(len(master_bedroom_weeks_list)):
    active_data = master_bedroom_weeks_list[i]
    active_prediction = order_0_predictions[i]
    active_prediction[active_prediction < 0.5] = 0
    active_prediction[active_prediction >= 0.5] = 1
    error = np.sum(np.abs(active_data - active_prediction)) / len(active_data)
    accuracy = 1 - error
    accuracies_order_0.append(accuracy)

for j in range(len(master_bedroom_weeks_list)):
    active_data = master_bedroom_weeks_list[j]
    active_prediction = order_5_predictions[j]
    active_prediction[active_prediction < 0.5] = 0
    active_prediction[active_prediction >= 0.5] = 1
    error = np.sum(np.abs(active_data - active_prediction)) / len(active_data)
    accuracy = 1 - error
    accuracies_order_5.append(accuracy)

for k in range(len(master_bedroom_weeks_list)):
    active_data = master_bedroom_weeks_list[k]
    active_prediction = order_1_predictions[k]
    active_prediction[active_prediction < 0.5] = 0
    active_prediction[active_prediction >= 0.5] = 1
    error = np.sum(np.abs(active_data - active_prediction)) / len(active_data)
    accuracy = 1 - error
    accuracies_order_1.append(accuracy)


# To-do: compare order_x_predictions[i] and order_x_predictions[y], should match or not???
print("Test for equality order 0 : ", np.array_equal(order_0_predictions[1], order_0_predictions[4]))
print("Test for equality order 5 : ", np.array_equal(order_5_predictions[1], order_5_predictions[4]))

# Creating dummy matrix for heatmap visualization
matrix = np.zeros((3, len(accuracies_order_5)))
matrix[0, :] = accuracies_order_0
matrix[1, :] = accuracies_order_1
matrix[2, :] = accuracies_order_5

plt.figure()
plt.subplot(311)
plt.plot(accuracies_order_0)
plt.subplot(312)
plt.plot(accuracies_order_1)
plt.subplot(313)
plt.plot(accuracies_order_5)
plt.show()
# Plot the accuracy
fig, ax = plt.subplots(figsize=(16, 16), facecolor='w', edgecolor='k')
ax = sns.heatmap(matrix, annot=True, vmax=1.0, vmin=0, cbar=True, fmt='.4g', ax=ax)
plt.title("Prediction accuracies for 16 consecutive weeks based on model trained with data of 10 weeks")
plt.xlabel("Week number")
plt.ylabel("Prediction model order [0, 1, 5])")
plt.show()
# The above shown result don't make any sense
order_5_predictions_week_0 = order_5_predictions[0]
order_5_predictions_week_1 = order_5_predictions[1]
order_5_predictions_week_2 = order_5_predictions[2]

order_5_predictions_week_0[order_5_predictions_week_0 < 0.5] = 0
order_5_predictions_week_0[order_5_predictions_week_0 >= 0.5] = 1
order_5_predictions_week_1[order_5_predictions_week_1 < 0.5] = 0
order_5_predictions_week_1[order_5_predictions_week_1 >= 0.5] = 1
order_5_predictions_week_2[order_5_predictions_week_2 < 0.5] = 0
order_5_predictions_week_2[order_5_predictions_week_2 >= 0.5] = 1

plt.figure()
plt.subplot(2, 3, 1)
plt.plot(master_bedroom_weeks_list[0])
plt.subplot(2, 3, 4)
plt.plot(order_5_predictions_week_0)
plt.subplot(2, 3, 2)
plt.plot(master_bedroom_weeks_list[1])
plt.subplot(2, 3, 5)
plt.plot(order_5_predictions_week_1)
plt.subplot(2, 3, 3)
plt.plot(master_bedroom_weeks_list[2])
plt.subplot(2, 3, 6)
plt.plot(order_5_predictions_week_2)
plt.show()

print("------------------")
print(np.array_equal(master_bedroom_weeks_list[0], master_bedroom_weeks_list[1]))
print("Accuracy week 0 : ", 1 - (np.sum(np.abs(master_bedroom_weeks_list[0] - order_5_predictions_week_0))) / len(order_5_predictions_week_0))
print("Accuracy week 1 : ", 1 - (np.sum(np.abs(master_bedroom_weeks_list[1] - order_5_predictions_week_1))) / len(order_5_predictions_week_1))
print("Accuracy week 2 : ", 1 - (np.sum(np.abs(master_bedroom_weeks_list[2] - order_5_predictions_week_2))) / len(order_5_predictions_week_2))
