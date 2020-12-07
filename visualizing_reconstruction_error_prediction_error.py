import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Set script-wide font size
plt.rc("font", size=11, family="Helvetica")
params = {'legend.fontsize': 10 }
plt.rcParams.update(params)

aruba_dataset = np.loadtxt("/home/adrian/Desktop/locations.min")

# times_array needs to be adjusted to the specified interval length
times_array_one_week = np.linspace(60, 604800, num=10080)

# Choose the desired room due to the numbers in locations.min
room = np.copy(aruba_dataset)
room[aruba_dataset == 0] = 1
room[aruba_dataset != 0] = 0

# load the predictions based on 12 weeks training, 4 weeks evaluating

room_prediction_order_0 = np.loadtxt("/home/adrian/fremen_predictions/aruba/binary/interval_60/master_bedroom_order_0.txt")
room_prediction_order_5 = np.loadtxt("/home/adrian/fremen_predictions/aruba/binary/interval_60/master_bedroom_order_5.txt")
room_prediction_order_10 = np.loadtxt("/home/adrian/fremen_predictions/aruba/binary/interval_60/master_bedroom_order_10.txt")

# Split the data into the separate weeks, isolate the first week and last week, consider interval length
room_weeks = np.split(room, 16)
room_first_week = room_weeks[0]
room_last_week = room_weeks[-1]

# Split the predictions into the separate weeks, isolate the first week and last week
room_prediction_order_0_first_week = np.split(room_prediction_order_0, 16)[0]
room_prediction_order_5_first_week = np.split(room_prediction_order_5, 16)[0]
room_prediction_order_10_first_week = np.split(room_prediction_order_10, 16)[0]

# Create binary predictions with threshold
room_prediction_order_0_first_week_thresholded = np.copy(room_prediction_order_0_first_week)
room_prediction_order_0_first_week_thresholded[room_prediction_order_0_first_week >= 0.5] = 1
room_prediction_order_0_first_week_thresholded[room_prediction_order_0_first_week < 0.5] = 0

room_prediction_order_5_first_week_thresholded = np.copy(room_prediction_order_5_first_week)
room_prediction_order_5_first_week_thresholded[room_prediction_order_5_first_week >= 0.5] = 1
room_prediction_order_5_first_week_thresholded[room_prediction_order_5_first_week < 0.5] = 0

room_prediction_order_10_first_week_thresholded = np.copy(room_prediction_order_10_first_week)
room_prediction_order_10_first_week_thresholded[room_prediction_order_10_first_week >= 0.5] = 1
room_prediction_order_10_first_week_thresholded[room_prediction_order_10_first_week < 0.5] = 0

# Get the errors over prediction orders up to 20 from the fremenserver
errors_prediction = np.array([0.3191468417644501, 0.15753968060016632, 0.11096230149269104, 0.1155753955245018, 0.11205357313156128, 0.11165674775838852, 0.1116071417927742, 0.112971231341362, 0.11324404925107956, 0.1134672611951828, 0.1130952388048172, 0.1132192462682724, 0.11356647312641144, 0.11530257761478424, 0.11577381193637848, 0.11614583432674408, 0.11688987910747528, 0.11684028059244156, 0.11721230298280716, 0.11783234030008316, 0.11785714328289032])
errors_reconstruction = np.array([0.38577795028686523, 0.11548149585723877, 0.07982949167490005, 0.08215462416410446, 0.08021701127290726, 0.07731059938669205, 0.07692307978868484, 0.07585739344358444, 0.07324162125587463, 0.07217593491077423, 0.07062584906816483, 0.07023832947015762, 0.06830071657896042, 0.06636311113834381, 0.0648130252957344, 0.0630691722035408, 0.0619066096842289, 0.061228446662425995, 0.06016276031732559, 0.06055028364062309, 0.05987212061882019])

fig, axs = plt.subplots(4, 1, sharex=True, sharey='col')
axs[0].plot(times_array_one_week, room_first_week, color='green')
axs[0].set_ylabel("Person occurrences")
axs[1].plot(times_array_one_week, room_prediction_order_0_first_week, color='green', label="$p(t)$")
axs[1].set_ylabel("Person occurrences")
axs[1].plot(times_array_one_week, room_prediction_order_0_first_week_thresholded, color='red', linestyle='dashed', label="s'(t)")
axs[1].legend(loc="upper right")
axs[2].plot(times_array_one_week, room_prediction_order_5_first_week, color='green', label="$p(t)$")
axs[2].set_ylabel("Person occurrences")
axs[2].plot(times_array_one_week, room_prediction_order_5_first_week_thresholded, color='red', linestyle='dashed', label="s'(t)")
axs[2].legend(loc="upper right")
axs[3].plot(times_array_one_week, room_prediction_order_10_first_week, color='green', label="$p(t)$")
axs[3].set_ylabel("Person occurrences")
axs[3].set_xlabel("Time $t$ [s]")
axs[3].plot(times_array_one_week, room_prediction_order_10_first_week_thresholded, color='red', linestyle='dashed', label="s'(t)")
axs[3].legend(loc="upper right")

plt.show()

# Plot reconstruction vs prediction error over model orders

plt.figure()
plt.plot(errors_reconstruction, color='blue', linestyle='solid', label="$\epsilon_r$")
plt.plot(errors_prediction, color='red', linestyle='dashed', label="$\epsilon_p$")
plt.ylabel("Prediction error")
plt.xlabel("Model order")
plt.xticks([0, 5, 10, 15, 20])
plt.legend(loc="upper right")
plt.show()






