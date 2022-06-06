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

class Replacing():

  # replace redundant letters
  def replacer(text):
      replace_list = """ሐ ሑ ሒ ሓ ሔ ሕ ሖ ጸ ጹ ጺ ጻ ጼ ጽ ጾ ኰ ኲ ጿ ኸ""".split(" ")
      ph = """ሀ ሁ ሂ ሀ ሄ ህ ሆ ፀ ፁ ፂ ፃ ፄ ፅ ፆ ኮ ኳ ፇ ኧ""".split(" ")
      for l in range(len(replace_list)):
        text = text.replace(replace_list[l], ph[l])
      return text