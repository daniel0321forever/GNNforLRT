import torch
import os
import shutil

dataset_dir  = "."
paths = os.listdir(dataset_dir)

# os.mkdir("../backup")
for path in paths:
    if path.split(".")[1] != "pyg":
        continue
    
    if int(path.split(".")[0]) <= 99:
        continue
    
    print(path, end=":")
    try:
        a = torch.load(path)
        print("non-empty file")

    except EOFError:
        print("empty file:")
#        shutil.move(path, "../backup")
    except RuntimeError as e:
        print("damaged file")
#        shutil.move(path, "../backup")
