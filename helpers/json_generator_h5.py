import numpy as np
import os
import json
import random
import h5py
import datetime 


# For Record Keeping and model generation
now = datetime.datetime.now()
run_uid = now.strftime("%Y_%m_%d_%H_%M")



img_per_movie = 245
pre_post_frame = 30
proportion_train = 80
proportion_test = (100-proportion_train)/2

total_movies = 168

data_dir = 'data/data_h5'
output_dir = 'data'

def save_json_list(list_all, list_selected, json_path):
    train_data = {}
    for i in list_selected:
        local_dict={}
        with h5py.File(list_all[i]+".h5",'r') as hf:
          local_data = np.array(hf['data'])
        local_data = local_data[5:]

        length_array=local_data.shape[0]
        list_pull = np.random.randint(low = pre_post_frame, high = length_array-pre_post_frame, size=img_per_movie)
        local_dict['frames']=list_pull.tolist()
        local_dict['path']= list_all[i]+".h5"
        local_dict['mean']=np.mean(local_data.flatten())
        local_dict['std']=np.std(local_data.flatten())
        
        train_data[str(i+1)] = local_dict
        
    with open(json_path, 'w+') as json_file:
        json.dump(train_data, json_file)

            


list_data = {}


#hard code path list for now


for i in np.arange(total_movies):
    list_data[i] = os.path.join(data_dir, str(i+1))
    
list_id = list(list_data.keys())

random.shuffle(list_id)

for i in np.arange(total_movies):
  
  list_data[i] = os.path.join(data_dir, str(i+1))


number_train_movies = int(np.round(proportion_train*float(total_movies)/100))
number_val_movies = int(np.round(proportion_test*float(total_movies)/100))
number_test_movies = number_val_movies
list_train=list_id[:number_train_movies]
list_after_train = np.setdiff1d(list_id, list_train)
random.shuffle(list_after_train)
list_val = list_after_train[:number_val_movies]
list_after_val = np.setdiff1d(list_after_train, list_val)
random.shuffle(list_after_val)
list_test = list_after_val[:number_test_movies]


local_path_train = os.path.join(output_dir, 'train.json')
local_path_val = os.path.join(output_dir, 'val.json')
local_path_test = os.path.join(output_dir, 'test.json')

print("Training sequence: " + str(list_train))
print("Validation sequence: " + str(list_val))
print("Testing sequence: " + str(list_test))

for i in list_id:
    print(list_data[i])

save_json_list(list_data, list_train, local_path_train)
save_json_list(list_data, list_val, local_path_val)
save_json_list(list_data, list_test, local_path_test)
