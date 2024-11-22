from inference import Inference
from glob import glob

# dir_path = '/data/ephemeral/home/Dongjin/level2-cv-semanticsegmentation-cv-02-lv3/Baseline/Dongjin/transformers_1120/trained_models/cont'
# model_dir_paths = glob(dir_path + '/*')
# model_dir_paths = sorted(model_dir_paths)

model_dir_paths = ['/data/ephemeral/home/Dongjin/level2-cv-semanticsegmentation-cv-02-lv3/Baseline/Dongjin/transformers_1120/trained_models/cont/upernet-convnext-small_cont_size_1024_cont_weight',
                   '/data/ephemeral/home/Dongjin/level2-cv-semanticsegmentation-cv-02-lv3/Baseline/Dongjin/transformers_1120/trained_models/cont/mit-b2_cont_size_1024_cont_weight']

for model_dir_path in model_dir_paths:
    inference = Inference(model_dir_path)
    inference.inference_and_save(mode='train')
    inference.inference_and_save(mode='valid')
    inference.inference_and_save(mode='test')
    
