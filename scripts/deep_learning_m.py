import os
import shutil

class Separating():

  def train_valid(src,dest_train,dest_valid):
    i = 0
    # fetch all files
    for file_name in os.listdir(src):
        # construct full file path  
        if i < ((len(os.listdir(src)))*0.8):
          source_train = src + file_name
          destination_train = dest_train + file_name
          # copy only files
          if os.path.isfile(source_train):
            shutil.copy(source_train, destination_train)
        else:
          source_valid = src + file_name
          destination_valid = dest_valid + file_name
            # copy only files
          if os.path.isfile(source_valid):
                shutil.copy(source_valid, destination_valid)
        i+=1