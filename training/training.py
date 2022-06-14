import numpy as np
from PIL import Image
import cv2
import os
import sys
import tensorflow as tf
import deepinterpolation as de
from shutil import copyfile
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime 
from typing import Any, Dict
import pathlib
import h5py
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

    

if __name__ == "__main__":
    now = datetime.datetime.now()
    run_uid = now.strftime("%Y_%m_%d_%H_%M")
    
    #Define Generator
    # MultiContinuousTifGenerator for .tif files / MovieJSONGenerator for .h5 files
    generator = "MovieJSONGenerator"

    # Change this to the location of training data
    train_path = \
        "/content/drive/MyDrive/Colab_Notebooks/train.json"

    val_path = "/content/drive/MyDrive/Colab_Notebooks/val.json"

    model_path = \
        "/content/drive/MyDrive/Colab_Notebooks/models"


    #Define steps per epoch
    steps_per_epoch = 7

    #Initiate meta-parameters objects
    training_param = {}
    generator_param = {}
    network_param = {}
    generator_test_param = {}



    #define pre_post_frame
    pre_post_frame = pre_post_frame

    #define start frame end frame
    start_frame = 0
    end_frame = -1

    batch_size = 35

    # Parameters used for validation test
    generator_test_param["type"] = "generator"
    generator_test_param["name"] = generator
    generator_test_param["pre_post_frame"] = pre_post_frame

    generator_test_param["train_path"] = val_path

    generator_test_param["batch_size"] = batch_size

    #Comment out if multiple files
    # generator_test_param["start_frame"] = start_frame
    # generator_test_param["end_frame"] = end_frame
    # generator_test_param["pre_post_omission"] = 0


    #Deactivate testing of epochs, give value -1
    generator_test_param["steps_per_epoch"] = -1


    # Parameters used for main data 
    generator_param["type"] = "generator"
    generator_param["steps_per_epoch"] = steps_per_epoch
    generator_param["name"] = generator
    generator_param["pre_post_frame"] = pre_post_frame

    # Change this to the location of training data
    generator_param["train_path"] = train_path
    generator_param["batch_size"] = batch_size

    #Comment out if multiple files
    # generator_param["start_frame"] = start_frame
    # generator_param["end_frame"] = end_frame

    # generator_param["pre_post_omission"] = 1


    # Network Topology
    network_param["type"] = "network"
    network_param["name"] = "unet_single_1024"


    # Parameters used for training
    training_param["type"] = "trainer"
    training_param["name"] = "core_trainer"
    training_param["run_uid"] = run_uid
    training_param["batch_size"] = generator_test_param["batch_size"]
    training_param["steps_per_epoch"] = steps_per_epoch

    #network model is periodically saved during training in between epochs
    training_param["period_save"] = 100
    training_param["nb_gpus"] = 1
    training_param["apply_learning_decay"] = 0
    training_param["nb_times_through_data"] = 1
    training_param["learning_rate"] = 0.0001
    training_param["pre_post_frame"] = pre_post_frame
    training_param["loss"] = "mean_absolute_error"
    #To enable multiple threads make larger than 1
    training_param["nb_workers"] = 1

    training_param["use_multiprocessing"] = False

    # training_param["cache_validation"] = False

    training_param["model_string"] = (
        network_param["name"]
        + "_"
        + training_param["loss"]
        + "_"
        + training_param["run_uid"]
    )

    jobdir = os.path.join(
            model_path,
            training_param["model_string"] + "_" + run_uid
        )
    training_param["output_dir"] = jobdir

    try: 
        os.mkdir(jobdir)
    except:
        print("folder already exists")

    path_training = os.path.join(jobdir, "training.json")
    json_obj = JsonSaver(training_param)
    json_obj.save_json(path_training)

    path_generator = os.path.join(jobdir, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    path_test_generator = os.path.join(jobdir, "test_generator.json")
    json_obj = JsonSaver(generator_test_param)
    json_obj.save_json(path_test_generator)

    path_network = os.path.join(jobdir, "network.json")
    json_obj = JsonSaver(network_param)
    json_obj.save_json(path_network)


    # Create all training objects
    generator_obj = ClassLoader(path_generator)
    generator_test_obj = ClassLoader(path_test_generator)

    network_obj = ClassLoader(path_network)

    trainer_obj = ClassLoader(path_training)

    train_generator = generator_obj.find_and_build()(path_generator)
    test_generator = generator_test_obj.find_and_build()(path_test_generator)

    network_callback = network_obj.find_and_build()(path_network)

    training_class = trainer_obj.find_and_build()(train_generator, test_generator, network_callback, path_training)

    training_class.run()

    training_class.finalize() 