import numpy as np
import matplotlib.pyplot as plt
from bson import ObjectId
from tqdm import tqdm
import pymongo

def add_grid():
    matrix = np.loadtxt("/home/adrian/counts_time_series_matrices/test.txt").reshape(2016, 60,
                                                                                     51)  # adro: need to get rid of hard coded shape numbers
    time_stamps = np.linspace(60 * 5, 604800, num=2016, endpoint=True)

    # Insert all cells of the grid into the database
    for row in tqdm(range(np.shape(matrix)[1]), desc="Row progress : "):
        for col in range(np.shape(matrix)[2]):
            cell_id = 'cell_row_' + str(row) + '_col_' + str(col)
            cell_time_stamps = time_stamps
            cell_states = matrix[:, row, col]

            insert_data(cell_id, cell_time_stamps, cell_states)
    print("All cells have been added to the database ...")

def insert_data(cell_id, cell_time_stamps, cell_states):
    cell_id = cell_id
    cell_time_stamps_converted = cell_time_stamps.tolist()
    cell_states_converted = cell_states.tolist()
    cell = {
        '_id': cell_id,
        'time_stamps': cell_time_stamps_converted,
        'states': cell_states_converted
    }
    collection.insert_one(cell)
    print("Cell has been added successfully")

def find_cell(cell_id):
    data = collection.find_one({'_id': cell_id})
    return data


if __name__ == '__main__':
   client = pymongo.MongoClient('localhost', 27017)
   print('Connection to database has been established.')
   db = client['uol_database']
   collection = db['cells']
   #add_grid()
   data = find_cell('cell_row_23_col_25')
   time_stamps = data['time_stamps']
   states = data['states']


   plt.plot(time_stamps, states)
   plt.title("Occupancy of the most occupied grid")
   plt.xlabel("Time [s]")
   plt.ylabel("Occupancy")
   plt.show()








   client.close()
