import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from bson import ObjectId
from tqdm import tqdm
import yaml

def insert_data(cell_id, cell_time_stamps, cell_states):
   cell = {
      '_id': cell_id,
      'time_stamps': cell_time_stamps,
      'states': cell_states
   }
   db.cells.insert_one(cell)

if __name__ == '__main__':

   with open("/home/adrian/Desktop/occupancy_grid_specifications.yaml") as f:
      dict_docs = yaml.load(f, Loader=yaml.FullLoader)
      print(dict_docs)
      exit()
   client = MongoClient('localhost', 27017)
   print('Connection to database has been established.')
   db = client['uol_database']
   collection = db['cells']
   data = []
   matrix = np.loadtxt("/home/adrian/counts_time_series_matrices/test.txt").reshape(2016, 60,51)  # adro: need to get rid of hard coded shape numbers
   time_stamps = np.linspace(60*5, 604800, num=2016, endpoint=True)
   # We want to insert all our cells into our database
   for row in tqdm(range(np.shape(matrix)[1]), desc="Row progress : "):
      for col in range(np.shape(matrix)[2]):
         cell_states = matrix[:, row, col]
         cell_time_stamps = time_stamps
         cell_id = 'cell_row_'+str(row)+'_col_'+str(col)
         insert_data(cell_id, cell_time_stamps, cell_states)

   print("Each cell of the grid has been added to the database...")
