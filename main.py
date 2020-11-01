import numpy as np
import os
from matplotlib import pyplot as plt


if __name__ == '__main__':
    print("Test has been started.")
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
    print("Program check statement")
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
    fig.suptitle("Room occupancy for one day")
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


    #plt.savefig("Room_occupancy_one_days.png")
    plt.show()

    # and now lets do everything for one week
    locations_one_week = locations_min[0:10080]
    print("Length of locations_one_week", len(locations_one_week))
    unique_week, counts_week = np.unique(locations_one_week, return_counts=True)
    dictionary_week = dict(zip(unique_week, counts_week))
    number_appearances_week = np.zeros(10)

    for room, appearances in dictionary_week.items():
        number_appearances_week[int(room)] = appearances

    # create an array for each room for one week, filled with 1 for occupancy and 0 for empty
    print("Pimmelberger")
    master_bedroom_week = np.zeros(len(locations_one_week))
    master_bathroom_week = np.zeros(len(locations_one_week))
    living_room_week = np.zeros(len(locations_one_week))
    kitchen_week = np.zeros(len(locations_one_week))
    center_week = np.zeros(len(locations_one_week))
    corridor_week = np.zeros(len(locations_one_week))
    second_bedroom_week = np.zeros(len(locations_one_week))
    office_week = np.zeros(len(locations_one_week))
    second_bathroom_week = np.zeros(len(locations_one_week))
    outside_week = np.zeros(len(locations_one_week))

    idx = 0
    for location in locations_one_week:
        location = int(location)
        print("Location")
        print(location)
        print("Index")
        print(idx)
        if location == 0:
            master_bedroom_week[idx] = 1
        elif location == 1:
            master_bathroom_week[idx] = 1
        elif location == 2:
            living_room_week[idx] = 1

        elif location == 3:
            kitchen_week[idx] = 1
        elif location == 4:
            center_week[idx] = 1
        elif location == 5:
            corridor_week[idx] = 1
        elif location == 6:
            second_bedroom_week[idx] = 1
        elif location == 7:
            office_week[idx] = 1
        elif location == 8:
            second_bathroom_week[idx] = 1
        elif location == 9:
            outside_week[idx] = 1
        else:
            print("Error")
        idx += 1

    print(office.shape)
    print(office)

    dummy_array = np.arange(10080)

    print("Was anything shown?")

    # lets try to create a plot for every room
    fig_week, axs_week = plt.subplots(5, 2)
    fig_week.suptitle('Room occupancy for one week')
    axs_week[0, 0].step(dummy_array, np.array(master_bedroom_week))
    axs_week[0, 0].set_title("Master bedroom")
    axs_week[0, 1].step(dummy_array, np.array(master_bathroom_week), 'tab:orange')
    axs_week[0, 1].set_title("Master bathroom")
    axs_week[1, 0].step(dummy_array, np.array(living_room_week), 'tab:green')
    axs_week[1, 0].set_title("Living room")
    axs_week[1, 1].step(dummy_array, np.array(kitchen_week), 'tab:red')
    axs_week[1, 1].set_title("Kitchen")
    axs_week[2, 0].step(dummy_array, np.array(center_week), 'tab:blue')
    axs_week[2, 0].set_title("Center")
    axs_week[2, 1].step(dummy_array, np.array(corridor_week), 'tab:blue')
    axs_week[2, 1].set_title("Corridor")
    axs_week[3, 0].step(dummy_array, np.array(second_bedroom_week), 'tab:pink')
    axs_week[3, 0].set_title("Second bedroom")
    axs_week[3, 1].step(dummy_array, np.array(office_week), 'tab:purple')
    axs_week[3, 1].set_title("Office")
    axs_week[4, 0].step(dummy_array, np.array(second_bathroom_week), 'tab:brown')
    axs_week[4, 0].set_title("Second bathroom")
    axs_week[4, 1].step(dummy_array, np.array(outside_week), 'tab:grey')
    axs_week[4, 1].set_title("Outside")

    for ax in axs_week.flat:
        ax.set(xlabel='Time stamps', ylabel='Occupancy')

    for ax in axs_week.flat:
        ax.label_outer()

    plt.savefig("Room_occupancy_one_week.png")
    plt.show()

    # saving the master_bedroom_day array to a text file for test purposes
    print("Does master_bedroom for one day still exist?")
    print(master_bedroom.shape)
    print(master_bedroom.size)
    np.savetxt('master_bedroom_one_day.out', master_bedroom)

    plt.plot(dummy_array, np.array(second_bedroom_week), 'tab:pink')
    plt.show()











