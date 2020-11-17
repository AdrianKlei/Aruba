import numpy as np
from matplotlib import pyplot as plt
import pymongo
import math

client = pymongo.MongoClient('localhost', 27017)
print('Connection to database has been established.')
db = client['uol_database_float']
collection = db['cell_recordings']

data = collection.find({"name":"cell_row_23_col_25"})
print(data[0])
print(data[1])

states = [entry["state"] for entry in data]
print(len(states))
weeks = np.array_split(states, 3)

cell_predictions = np.loadtxt("/home/adrian/Desktop/cell_row_23_col_25_frongo.txt")

cell_predictions = np.array_split(cell_predictions, 3)

plt.figure()
plt.plot(weeks[0])
plt.plot(cell_predictions[0])
plt.show()

# Calculate rmse
rmse = np.sqrt(np.sum(np.square(weeks[0] - cell_predictions[0])) / len(cell_predictions[0]))
print("rmse sums up to : ", rmse)