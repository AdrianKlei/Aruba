import numpy as np
import pymongo
import yaml
from tqdm import tqdm

aruba_dataset = np.loadtxt("/home/adrian/Desktop/locations.min")
master_bedroom_occupancy = np.copy(aruba_dataset)
master_bedroom_occupancy[aruba_dataset == 0] = 1
master_bedroom_occupancy[aruba_dataset != 0] = 0

living_room_occupancy = np.copy(aruba_dataset)
living_room_occupancy[aruba_dataset == 2] = 1
living_room_occupancy[aruba_dataset != 2] = 0

outside_occupancy = np.copy(aruba_dataset)
outside_occupancy[aruba_dataset == 9] = 1
outside_occupancy[aruba_dataset != 9] = 0

print(len(master_bedroom_occupancy))
time_stamps = np.linspace(60, 9676800, num=161280, endpoint=True)
time_stamps_converted = time_stamps.tolist()

master_bedroom_states = master_bedroom_occupancy
master_bedroom_states_converted = master_bedroom_states.tolist()

living_room_states = living_room_occupancy
living_room_states_converted = living_room_states.tolist()

outside_states = outside_occupancy
outside_states_converted = outside_states.tolist()
print("----------------------------------------")
print("Number of 1's in living room : ", np.count_nonzero(living_room_states[:120960] == 1))
print("Number of 0's in living room : ", np.count_nonzero(living_room_states[:120960] == 0))
print("Average living room occupancy : ", float(np.count_nonzero(living_room_states[:120960] == 1))/ float(len(living_room_states[:120960])))

exit()  # This statement prevents double data base entries
client = pymongo.MongoClient('localhost', 27017)
print("Connection to database has been established.")
db = client['aruba_database']
collection = db['cell_recordings']

for i in tqdm(range(len(time_stamps_converted)), desc="Cell progress : "):
    cell_1 = {
        'name': 'master_bedroom',
        'timestamp': time_stamps_converted[i],
        'state': master_bedroom_states_converted[i]
    }
    collection.insert_one(cell_1)

    cell_2 = {
        'name': 'living_room',
        'timestamp': time_stamps_converted[i],
        'state': living_room_states_converted[i]
    }
    collection.insert_one(cell_2)

    cell_3 = {
        'name': 'outside',
        'timestamp': time_stamps_converted[i],
        'state': outside_states_converted[i]
    }
    collection.insert_one(cell_3)
print("All cells have been added to the DB!")