import numpy as np
from matplotlib import pyplot as plt
import pymongo
import math

client = pymongo.MongoClient('localhost', 27017)
print('Connection to database has been established.')
db = client['aruba_database_float']
collection = db['cell_recordings']

data = collection.find({"name":"master_bedroom"})
print(data[0])
print(data[1])

states = [entry["state"] for entry in data]
print(len(states))
weeks = np.array_split(states, 16)

master_bedroom_predictions = np.loadtxt("/home/adrian/Desktop/master_bedroom_frongo.txt")

master_bedroom_predictions = np.array_split(master_bedroom_predictions, 16)

plt.figure()
plt.plot(weeks[0])
plt.plot(master_bedroom_predictions[0])
plt.show()

# Calculate rmse
rmse = np.sqrt(np.sum(np.square(weeks[0] - master_bedroom_predictions[0])) / len(master_bedroom_predictions[0]))
print("rmse sums up to : ", rmse)