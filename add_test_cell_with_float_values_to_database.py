import pymongo
import numpy as np
import math
from matplotlib import pyplot as plt

from tqdm import tqdm

aruba_dataset = np.loadtxt("/home/adrian/Desktop/locations.min")
master_bedroom = np.copy(aruba_dataset)
master_bedroom[aruba_dataset == 0] = 1
master_bedroom[aruba_dataset != 0] = 0

# Create bins of 10 mins to count the occupancies in the master bedroom during this interval
master_bedroom_occupancy = np.zeros(len(master_bedroom)/10)
times_array = np.linspace(600, 60*60*24*7*16, num=16128)
print(len(master_bedroom_occupancy))
print(len(times_array))

for i in range(len(master_bedroom_occupancy)):
    master_bedroom_occupancy[i] = sum(master_bedroom[i*10:(i+1)*10])


plt.figure()
plt.plot(times_array[:1008], master_bedroom_occupancy[:1008])
plt.title("Cumulated master bedroom occupancy for one week")
plt.show()

# Save the values to the database "aruba_database_float"
client = pymongo.MongoClient('localhost', 27017)
print("Connection to database has been established.")
db = client['aruba_database_float']
collection = db['cell_recordings']
master_bedroom_converted = master_bedroom_occupancy.tolist()
times_array_converted = times_array.tolist()

for i in tqdm(range(len(master_bedroom_occupancy)), desc="Cell progress"):
    cell = {
        'name':'master_bedroom',
        'timestamp': times_array_converted[i],
        'state': master_bedroom_converted[i]
        }
    collection.insert_one(cell)

print("Finished...")