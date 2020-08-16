import numpy as np
import os
from matplotlib import pyplot as plt


if __name__ == '__main__':
    print("Pimmelberger")
    locations_min = np.loadtxt(fname="/home/adrian/Downloads/strands/aruba/locations.min")

    with open("/home/adrian/Downloads/strands/aruba/locations.names") as file_in:
        location_names = []
        for line in file_in:
            location_names.append(line)

    print("File has been read into the lines array")

    for line in location_names:
        print(line)

    print("The length of the data array is ", len(locations_min))
    print(type(locations_min))

    unique, counts = np.unique(locations_min, return_counts=True)
    dictionary = dict(zip(unique, counts))
    print(dictionary)

    rooms = np.array([0,1,2,3,4,5,6,7,8,9])
    number_appearances_total = np.zeros(10)
    #print(len(number_appearances))


    for room, appearances in dictionary.items():
     number_appearances_total[int(room)] = appearances

    # and now lets do everything for one day
    locations_one_day = locations_min[0:1440]
    print("Length of locations_one_day", len(locations_one_day))
    unique_day, counts_day = np.unique(locations_one_day, return_counts=True)
    dictionary_day = dict(zip(unique_day, counts_day))
    number_appearances_day = np.zeros(10)

    for room, appearances in dictionary_day.items():
        number_appearances_day[int(room)] = appearances


    # now we plot the total appearances and appearances for one day
    plt.figure(1)
    plt.subplot(211)
    plt.bar(rooms, number_appearances_total)
    plt.title("Person presence for 16 weeks")
    plt.ylabel("Occurence times")
    plt.xlabel("Rooms")
    plt.xticks(rooms, location_names, rotation=60, fontsize=8)

    plt.subplot(212)
    plt.bar(rooms, number_appearances_day)
    plt.title("Person presence for one day")
    plt.ylabel("Occurence times")
    plt.xlabel("Rooms")
    plt.xticks(rooms, location_names, rotation=60, fontsize=8)

    plt.show()

    # create an array for each room for one day, filled with 1 for occupancy and 0 for empty
    print("Pimmelberger")
    master_bedroom = np.zeros(len(locations_one_day))
    master_bathroom = np.zeros(len(locations_one_day))
    living_room = np.zeros(len(locations_one_day))
    kitchen = np.zeros(len(locations_one_day))
    center = np.zeros(len(locations_one_day))
    corridor = np.zeros(len(locations_one_day))
    second_bedroom = np.zeros(len(locations_one_day))
    office = np.zeros(len(locations_one_day))
    second_bathroom = np.zeros(len(locations_one_day))
    outside = np.zeros(len(locations_one_day))

    idx = 0
    for location in locations_one_day:
        location = int(location)
        print("Location")
        print(location)
        print("Index")
        print(idx)
        if location == 0:
            master_bedroom[idx] = 1
        elif location == 1:
            master_bathroom[idx] = 1
            # why do I get the information that 'statement seems to have no effect'?
        elif location == 2:
            living_room[idx] = 1

        elif location == 3:
            kitchen[idx] = 1
        elif location == 4:
            center[idx] = 1
        elif location == 5:
            corridor[idx] = 1
        elif location == 6:
            second_bedroom[idx] = 1
        elif location == 7:
            office[idx] = 1
        elif location == 8:
            second_bathroom[idx] = 1
        elif location == 9:
            outside[idx] = 1
        else:
            print("Error")
        idx += 1

    print(office.shape)
    print(office)

    dummy_array = np.arange(1440)
    #xaxis = np.arange(1440)
    #yaxis = np.array(master_bathroom)
    #plt.step(xaxis, yaxis)
    #plt.title("Person occurence in master bathroom")
    #plt.xlabel("Time stamps")
    #plt.ylabel("Occupancy")
    #plt.show()



    print("Was anything shown?")

    # lets try to create a plot for every room
    fig, axs = plt.subplots(5, 2)
    axs[0, 0].step(dummy_array, np.array(master_bedroom))
    axs[0, 0].set_title("Master bedroom")
    axs[0, 1].step(dummy_array, np.array(master_bathroom), 'tab:orange')
    axs[0, 1].set_title("Master bathroom")
    axs[1, 0].step(dummy_array, np.array(living_room), 'tab:green')
    axs[1, 0].set_title("Living room")
    axs[1, 1].step(dummy_array, np.array(kitchen), 'tab:red')
    axs[1, 1].set_title("Kitchen")
    axs[2, 0].step(dummy_array, np.array(center), 'tab:blue')
    axs[2, 0].set_title("Center")
    axs[2, 1].step(dummy_array, np.array(corridor), 'tab:blue')
    axs[2, 1].set_title("Corridor")
    axs[3, 0].step(dummy_array, np.array(second_bedroom), 'tab:pink')
    axs[3, 0].set_title("Second bedroom")
    axs[3, 1].step(dummy_array, np.array(office), 'tab:purple')
    axs[3, 1].set_title("Office")
    axs[4, 0].step(dummy_array, np.array(second_bathroom), 'tab:brown')
    axs[4, 0].set_title("Second bathroom")
    axs[4, 1].step(dummy_array, np.array(outside), 'tab:grey')
    axs[4, 1].set_title("Outside")

    for ax in axs.flat:
        ax.set(xlabel='Time stamps', ylabel='Occupancy')

    for ax in axs.flat:
        ax.label_outer()

    plt.show()













