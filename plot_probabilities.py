import numpy as np
from matplotlib import pyplot as plt

# Needs to be specified which folder and dataset is used
# Therefore dummy_variables aruba_dataset und uol_dataset

if __name__ == '__main__':
    aruba_dataset = False
    uol_dataset = True
    print("Script 'plot_probabilities.py' has started...")

    if aruba_dataset:
        requested_room = np.loadtxt("/home/adrian/requested_rooms/requested_room_states.txt")
        path = "/home/adrian/requested_rooms/fremen_model_probabilities/model_order_"

    if uol_dataset:
        sample_frequency = 15   # Sample frequency [minutes]
        requested_room = np.loadtxt("/media/adrian/Volume/occupancy_grid/cell_row_23_col_25/binary_states_array.txt")
        path = "/media/adrian/Volume/occupancy_grid/cell_row_23_col_25/fremen_model_probabilities/model_order_"


    times_array = np.linspace(sample_frequency*60, 604800.0, num=604800/(sample_frequency*60), endpoint=True)

    files_array = []
    for i in range(5):
        file = np.loadtxt(path + str(i) + ".txt")
        files_array.append(file)

    model_order_0 = files_array[0]
    model_order_1 = files_array[1]
    model_order_2 = files_array[2]
    model_order_3 = files_array[3]
    model_order_4 = files_array[4]

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(times_array, model_order_0)
    axs[0, 0].set_title('Model order 0')
    axs[0, 1].plot(times_array, model_order_1, "tab:orange")
    axs[0, 1].set_title('Model order 1')
    axs[1, 0].plot(times_array, model_order_2, "tab:green")
    axs[1, 0].set_title('Model order 2')
    axs[1, 1].plot(times_array, model_order_3, "tab:red")
    axs[1, 1].set_title('Model order 3')

    for ax in axs.flat:
        ax.set(xlabel="Time [s]", ylabel="Probability")

    plt.show()

    plt.figure(2)
    plt.subplot(241)
    plt.plot(times_array, model_order_0, 'tab:orange')
    plt.subplot(245)
    plt.plot(times_array, requested_room)
    plt.subplot(242)
    plt.plot(times_array, model_order_1, 'tab:green')
    plt.subplot(246)
    plt.plot(times_array, requested_room)
    plt.subplot(243)
    plt.plot(times_array, model_order_2, 'tab:red')
    plt.subplot(247)
    plt.plot(times_array, requested_room)
    plt.subplot(244)
    plt.plot(times_array, model_order_3, 'tab:purple')
    plt.subplot(248)
    plt.plot(times_array, requested_room)
    plt.show()

    exit() # Exit point for current state of uol_dataset=True

    # visualizing the prediction errors over the different orders
    prediction_errors = np.loadtxt("/home/adrian/requested_rooms/fremen_model_errors/errors.txt")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(len(prediction_errors))
    y = 100 * prediction_errors
    line, = ax.plot(x, y)
    ymin = min(y)
    ymax = max(y)
    xpos = np.argmin(y)
    xmin = x[xpos]

    ax.annotate('error_min: order ' + str(xmin) + ' error: ' + str(ymin), xy=(xmin, ymin), xytext=(xmin, ymin + 0.2*ymin),
                arrowprops=dict(facecolor='black', shrink=0.02),
                )
    plt.title("Outside: Prediction errors of different model orders")
    plt.ylabel("error [%]")
    plt.xlabel("order [n]")
    plt.show()


    print("Minimum error is produced by model order : ", np.argmin(prediction_errors))

    # check if the result of fremen-evaluate matches the following computation
    optimal_model_order = np.loadtxt("/home/adrian/requested_rooms/fremen_model_probabilities/model_order_" + str(xpos) + ".txt")
    optimal_model_order[optimal_model_order < 0.5] = 0
    optimal_model_order[optimal_model_order >= 0.5] = 1
    error = np.sum(np.abs(requested_room - optimal_model_order)) / len(optimal_model_order)
    print("The error sums up to : ", error)

    plt.figure(3)
    plt.subplot(211)
    plt.title("Outside: Prediction of model order " + str(xpos))
    plt.ylabel("Room occupancy")
    plt.xlabel("Time [s]")
    plt.plot(times_array, optimal_model_order)
    plt.subplot(212)
    plt.title("Original data")
    plt.ylabel("Room occupancy")
    plt.xlabel("Time [s]")
    plt.plot(times_array, requested_room)
    plt.show()

    number_of_ones = np.count_nonzero(requested_room == 1)
    number_of_zeros = np.count_nonzero(requested_room == 0)
    static_occupancy_probability = 100 * (float(number_of_ones) / (float(number_of_ones) + float(number_of_zeros)))
    print("Static probability for room occupancy is : " + str(static_occupancy_probability) + "%")

    # to do: plot the errors of every training model order for a test data of another week
    # therefore we need to create a binary array for every room for the second week
    # for clarity purposes this functionality should be outsourced to an external function

    locations_min = np.loadtxt(fname="/home/adrian/Downloads/strands/aruba/locations.min")
    locations_second_week = locations_min[20160:30240]
    print("Length of locations_one_week", len(locations_second_week))
    unique_week, counts_week = np.unique(locations_second_week, return_counts=True)
    dictionary_week = dict(zip(unique_week, counts_week))
    number_appearances_week = np.zeros(10)

    for room, appearances in dictionary_week.items():
        number_appearances_week[int(room)] = appearances

    # create an array for each room for one week, filled with 1 for occupancy and 0 for empty
    print("Pimmelberger")
    master_bedroom_second_week = np.zeros(len(locations_second_week))
    master_bathroom_second_week = np.zeros(len(locations_second_week))
    living_room_second_week = np.zeros(len(locations_second_week))
    kitchen_second_week = np.zeros(len(locations_second_week))
    center_second_week = np.zeros(len(locations_second_week))
    corridor_second_week = np.zeros(len(locations_second_week))
    second_bedroom_second_week = np.zeros(len(locations_second_week))
    office_second_week = np.zeros(len(locations_second_week))
    second_bathroom_second_week = np.zeros(len(locations_second_week))
    outside_second_week = np.zeros(len(locations_second_week))

    idx = 0
    for location in locations_second_week:
        location = int(location)
        print("Location")
        print(location)
        print("Index")
        print(idx)
        if location == 0:
            master_bedroom_second_week[idx] = 1
        elif location == 1:
            master_bathroom_second_week[idx] = 1
        elif location == 2:
            living_room_second_week[idx] = 1

        elif location == 3:
            kitchen_second_week[idx] = 1
        elif location == 4:
            center_second_week[idx] = 1
        elif location == 5:
            corridor_second_week[idx] = 1
        elif location == 6:
            second_bedroom_second_week[idx] = 1
        elif location == 7:
            office_second_week[idx] = 1
        elif location == 8:
            second_bathroom_second_week[idx] = 1
        elif location == 9:
            outside_second_week[idx] = 1
        else:
            print("Error")
        idx += 1

    # after we created the data array for every room for the second week, we have to compare these
    # occupancy values with the predictions from every model order from the training data of the first week
    # we calculate this for the master_bedroom

    prediction_arrays = []
    for i in range(100):
        prediction = np.loadtxt("/home/adrian/requested_rooms/fremen_model_probabilities/model_order_" + str(i) + ".txt")
        prediction_arrays.append(prediction)

    # change prediction_arrays to binary states with threshold of 0.5
    for i in range(len(prediction_arrays)):
        active_array = prediction_arrays[i]
        active_array[active_array < 0.5] = 0
        active_array[active_array >= 0.5] = 1

    # compare predicted states (trained with first week data) of every order with ground truth of desired room (second week)
    errors_array = []
    active_room = outside_second_week
    for i in range(len(prediction_arrays)):
        active_array = prediction_arrays[i]
        error = 100*np.sum(np.abs(active_room - active_array)) / len(active_room) # multiplicate with 100 to change format to percentage
        errors_array.append(error)

    orders = np.arange(len(errors_array))
    errors_min = min(errors_array)
    errors_min_idx = np.argmin(errors_array)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot(orders, errors_array)

    ax.annotate('error_min: order ' + str(errors_min_idx) + ' error: ' + str(errors_min), xy=(errors_min_idx, errors_min), xytext=(errors_min_idx, errors_min + 0.2 * errors_min),
                arrowprops=dict(facecolor='black', shrink=0.02),
                )
    plt.title("Outside: Test error of different model orders")
    plt.ylabel("error [%]")
    plt.xlabel("order [n]")
    plt.show()










