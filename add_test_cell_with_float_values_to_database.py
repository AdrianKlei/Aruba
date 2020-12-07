import pymongo
import numpy as np
import math
from matplotlib import pyplot as plt

from tqdm import tqdm
fileObj = open("/home/adrian/Downloads/strands/aruba/locations.names", "r") #opens the file in read mode
words = fileObj.read().splitlines() #puts the file into an array
fileObj.close()

for word in words:
    print(word)

aruba_dataset = np.loadtxt("/home/adrian/Desktop/locations.min")
for i in range(len(words)):

    room_occupancy = np.copy(aruba_dataset)
    room_occupancy[aruba_dataset == i] = 1
    room_occupancy[aruba_dataset != i] = 0

    # Create bins of 10 mins to count the occupancies in the master bedroom during this interval
    room_occupancy_acc = np.zeros(len(room_occupancy)/10)
    # The times_array specifies how many time intervals we have
    times_array = np.linspace(600, 60*60*24*7*16, num=16128)
    print(len(room_occupancy_acc))
    print(len(times_array))

    for j in range(len(room_occupancy_acc)):
        room_occupancy_acc[j] = sum(room_occupancy[j*10:(j+1)*10])


    #plt.figure()
    #plt.plot(times_array[:1008], room_occupancy_acc[:1008])
    #plt.title("Cumulated room occupancy for one week")
    #plt.show()

    # Save the values to the database "aruba_database_float"
    client = pymongo.MongoClient('localhost', 27017)
    print("Connection to database has been established.")
    db = client['aruba_database_float']
    collection = db['cell_recordings']
    room_occupancy_converted = room_occupancy_acc.tolist()
    times_array_converted = times_array.tolist()

    for k in tqdm(range(len(room_occupancy_acc)), desc="Cell progress"):
        cell = {
            'name': "cell_row_0_col_" + str(i),
            'timestamp': times_array_converted[k],
            'state': room_occupancy_converted[k]
            }
        collection.insert_one(cell)

    print("Finished...")

