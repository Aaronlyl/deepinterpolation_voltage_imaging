from deepinterpolation.generic import JsonSaver, ClassLoader
import os
import numpy as np


def infer(train_path, output_path, model_path, jobdir):
    generator_param = {}
    inferrence_param = {}


    start_frame = 5
    end_frame = -1
    pre_post_frame = 30

    batch_size = 35

    # We are reusing the data generator for training here.
    generator_param["type"] = "generator"
    generator_param["name"] = "OphysGenerator"
    generator_param["pre_post_frame"] = pre_post_frame
    generator_param["steps_per_epoch"] = -1
    # No steps necessary for inference as epochs are not relevant.
    # -1 deactivate it.



    #Change this to where to store output
    inferrence_param["output_file"] = output_path

    generator_param["train_path"] = train_path
    generator_param["batch_size"] = batch_size


    generator_param["start_frame"] = start_frame
    generator_param["end_frame"] = end_frame  # -1 to go until the end.
    generator_param[
        "randomize"
    ] = 0
    # This is important to keep the order and avoid the
    # randomization used during training

    inferrence_param["type"] = "inferrence"
    inferrence_param["name"] = "core_inferrence"



    inferrence_param["model_path"] = model_path



    try:
        os.mkdir(jobdir)
    except Exception:
        print("folder already exists")

    path_generator = os.path.join(jobdir, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    path_infer = os.path.join(jobdir, "inferrence.json")
    json_obj = JsonSaver(inferrence_param)
    json_obj.save_json(path_infer)

    generator_obj = ClassLoader(path_generator)
    data_generator = generator_obj.find_and_build()(path_generator)

    inferrence_obj = ClassLoader(path_infer)
    inferrence_class = inferrence_obj.find_and_build()(path_infer,
                                                        data_generator)

    # Except this to be slow on a laptop without GPU. Inference needs
    # parallelization to be effective.
    inferrence_class.run()


nums = [7, 27, 28, 29, 50, 60, 64, 83, 100, 114, 118]

train_paths = []
output_paths = []

model_path = \
    "models/unet_single_1024_mean_absolute_error_2022_06_11_08_07_2022_06_11_08_07/2022_06_11_08_07_unet_single_1024_mean_absolute_error_2022_06_11_08_07_model.h5"
    
    
up_train_path = "../data/data_h5/"
up_output_path = "../data/data_denoised/"

jobdir = '..data/data_deniosed/'

for i in nums:
    train_paths.append(up_train_path+str(i+1)+".h5")
    output_paths.append(up_output_path+str(i+1)+"_denoised.h5")

    
for i in range(len(train_paths)):
  infer(train_paths[i], output_paths[i], model_path, jobdir)