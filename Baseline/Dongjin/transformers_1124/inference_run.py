from glob import glob
from inference import Inference

dir_path = 'trained_models'
dir1_paths = glob(dir_path + '/**')

for dir1_path in dir1_paths:
    dir2_paths = glob(dir1_path + '/**')

    for dir2_path in dir2_paths:        
        print(dir2_path)
        inference = Inference(dir2_path)
        inference.inference_and_save(mode='valid')
        inference.inference_and_save(mode='test')