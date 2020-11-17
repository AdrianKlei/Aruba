import numpy as np
import pymongo
# Important : don't save np.arrays to database, convert it into arche-datatyp before or find another solution

if __name__ == '__main__':
    client = pymongo.MongoClient('localhost', 27017)
    print("Connection to database has been established.")
    db = client['test_cell_database']
    collection = db['cell_recordings']

    cell_recording = {
        'name' : 'cell_row_23_col_25',
        'timestamp' : 5,
        'state' : 5

    }
    # We need 400 timestamps of the test cell over a period of 4 weeks
    # 100 timestamps per week and 1 week has 604800 seconds
    # One timestamp every 6048 seconds

    timestamps = np.linspace(6048, 2419200, num=400, endpoint=True)
    states = np.zeros(400,)

    states[:100] = 1
    states[300:] = 1
    timestamps_converted = timestamps.tolist()
    states_converted = states.tolist()

    # Add each timestamp to the database
    for i in range(len(timestamps_converted)):
        cell_recording = {
            'name': 'cell_row_23_col_25',
            'timestamp': timestamps_converted[i],
            'state': states_converted[i]

        }
        collection.insert_one(cell_recording)
    print("All records of the test cell have been added to the database.")

