

import numpy as np
import os
import tifffile
import h5py




def save_file(img, out_path, Fd=None, F0=None, activation_map=None, tSNR=None):
    try:
         with h5py.File(out_path, 'w') as hf:
            hf.create_dataset("data",  data=img)
            if Fd is not None:
                hf.create_dataset("Fd", data=Fd)
            if F0 is not None:
                hf.create_dataset("F0", data=F0)
            if activation_map is not None:
                hf.create_dataset("act_map", data=activation_map)
            if tSNR is not None:
                hf.create_dataset("tSNR", data=tSNR)
            print(("file saved: " + out_path))
    except:
        print("error saving file.")


def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[1] // new_shape[0],
             new_shape[1], arr.shape[2] // new_shape[1])
    img = np.zeros((arr.shape[0],new_shape[0],new_shape[1]))
    for t in range(arr.shape[0]):
        img[t,:,:] = arr[t,:,:].reshape(shape).mean(-1).mean(1)
    
    return img


def crop(img, x_dim, y_dim, start_skip=0):

    img_cropped = np.zeros((img.shape[0]-start_skip, int(x_dim),int(y_dim))).astype(np.uint16)
    c = int(img.shape[1]/2.0)
    for i in np.arange(start_skip,img.shape[0]):
        img_cropped[i-start_skip,:,:] = img[i,int(c-x_dim/2):int(c+x_dim/2),int(c-y_dim/2):int(c+y_dim/2)]
    print(img_cropped.shape)

    return img_cropped


def get_F0(img):
    return np.mean(img[0:9])


def get_Fd(img):
    Fd = np.mean(np.mean(img[:,0:19,0:19],axis=0))
    return Fd


def process(img_path, out_path):
    
    img = tifffile.imread(img_path)
    img_res = rebin(img, [256,256])
    Fd = get_Fd(img_res)
    F0 = get_F0(img_crop)
    img_crop = crop(img_res, 128, 128)
    print(img_crop.shape)
    save_file(img_crop, out_path, F0=F0, Fd=Fd)
    
if __name__== "__main__":
    
    img_paths = []

    out_paths = []

    in_dir = "/rds/general/user/yll3918/projects/thefarm2/live/Fdirefly/VoltageImaging"
    out_dir = "data/data_h5"


    img_paths.append(os.path.join(in_dir, "/20191111/slice1/cell1/no_MLA/5_repeats_100mA/5_no_MLA_100Hz_100mA_long_stim_corr2_1/5_no_MLA_100Hz_100mA_long_stim_corr2_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/1.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice1/cell1/no_MLA/5_repeats_100mA/5_no_MLA_100Hz_100mA_long_stim_corr2_2/5_no_MLA_100Hz_100mA_long_stim_corr2_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/2.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice1/cell1/no_MLA/5_repeats_100mA/5_no_MLA_100Hz_100mA_long_stim_corr2_3/5_no_MLA_100Hz_100mA_long_stim_corr2_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/3.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice1/cell1/no_MLA/5_repeats_100mA/5_no_MLA_100Hz_100mA_long_stim_corr2_4/5_no_MLA_100Hz_100mA_long_stim_corr2_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/4.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice1/cell2/no-MLA/5_repeats_100ma/5_no_MLA_100Hz_100mA_long_stim_1/5_no_MLA_100Hz_100mA_long_stim_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/5.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice1/cell2/no-MLA/5_repeats_100ma/5_no_MLA_100Hz_100mA_long_stim_2/5_no_MLA_100Hz_100mA_long_stim_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/6.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice1/cell2/no-MLA/5_repeats_100ma/5_no_MLA_100Hz_100mA_long_stim_3/5_no_MLA_100Hz_100mA_long_stim_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/7.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice1/cell2/no-MLA/5_repeats_100ma/5_no_MLA_100Hz_100mA_long_stim_4/5_no_MLA_100Hz_100mA_long_stim_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/8.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice3/cell1/no-MLA/10_repeats_200mA/10_no-MLA_100Hz_200mA_long-stim_1/10_no-MLA_100Hz_200mA_long-stim_1_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/9.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice2/cell1/no-MLA/5_repeats_100mA-corr/5_no-MLA_100Hz_100mA_long-stim_1/5_no-MLA_100Hz_100mA_long-stim_1_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/10.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice2/cell1/no-MLA/5_repeats_100mA-corr/5_no-MLA_100Hz_100mA_long-stim_2/5_no-MLA_100Hz_100mA_long-stim_2_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/11.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice2/cell1/no-MLA/5_repeats_100mA-corr/5_no-MLA_100Hz_100mA_long-stim_3/5_no-MLA_100Hz_100mA_long-stim_3_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/12.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice2/cell1/no-MLA/5_repeats_100mA-corr/5_no-MLA_100Hz_100mA_long-stim_4/5_no-MLA_100Hz_100mA_long-stim_4_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/13.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice2/cell1/no-MLA/5_repeats_100mA-corr/5_no-MLA_100Hz_100mA_long-stim_5/5_no-MLA_100Hz_100mA_long-stim_5_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/14.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice3/cell1/no-MLA/10_repeats_200mA/10_no-MLA_100Hz_200mA_long-stim_3/10_no-MLA_100Hz_200mA_long-stim_3_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/15.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice3/cell1/no-MLA/10_repeats_200mA/10_no-MLA_100Hz_200mA_long-stim_4/10_no-MLA_100Hz_200mA_long-stim_4_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/16.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice3/cell1/no-MLA/10_repeats_200mA/10_no-MLA_100Hz_200mA_long-stim_5/10_no-MLA_100Hz_200mA_long-stim_5_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/17.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice3/cell1/no-MLA/10_repeats_200mA/10_no-MLA_100Hz_200mA_long-stim_6/10_no-MLA_100Hz_200mA_long-stim_6_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/18.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice3/cell1/no-MLA/10_repeats_200mA/10_no-MLA_100Hz_200mA_long-stim_7/10_no-MLA_100Hz_200mA_long-stim_7_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/19.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice3/cell1/no-MLA/10_repeats_200mA/10_no-MLA_100Hz_200mA_long-stim_8/10_no-MLA_100Hz_200mA_long-stim_8_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/20.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice3/cell1/no-MLA/10_repeats_200mA/10_no-MLA_100Hz_200mA_long-stim_9/10_no-MLA_100Hz_200mA_long-stim_9_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/21.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice3/cell1/no-MLA/10_repeats_200mA/10_no-MLA_100Hz_200mA_long-stim_10/10_no-MLA_100Hz_200mA_long-stim_10_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/22.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice4/cell1/no-MLA/5_long-stim-200mA/5_no-MLA_100Hz_200mA_long-stim_1/5_no-MLA_100Hz_200mA_long-stim_1_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/23.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice4/cell1/no-MLA/5_long-stim-200mA/5_no-MLA_100Hz_200mA_long-stim_2/5_no-MLA_100Hz_200mA_long-stim_2_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/24.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice4/cell1/no-MLA/5_long-stim-200mA/5_no-MLA_100Hz_200mA_long-stim_3/5_no-MLA_100Hz_200mA_long-stim_3_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/25.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice4/cell1/no-MLA/5_long-stim-200mA/5_no-MLA_100Hz_200mA_long-stim_4/5_no-MLA_100Hz_200mA_long-stim_4_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/26.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice4/cell1/no-MLA/5_long-stim-200mA/5_no-MLA_100Hz_200mA_long-stim_5/5_no-MLA_100Hz_200mA_long-stim_5_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/27.h5"))

    img_paths.append(os.path.join(in_dir, "/20191111/slice4/cell1/no-MLA/5_long-stim-200mA/5_no-MLA_100Hz_200mA_long-stim_6/5_no-MLA_100Hz_200mA_long-stim_6_MMStack_Default.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/28.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice1/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_1/1x1_f_28_50_mA_p_stim_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/29.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice1/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_2/1x1_f_28_50_mA_p_stim_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/30.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice1/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_3/1x1_f_28_50_mA_p_stim_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/31.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice1/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_4/1x1_f_28_50_mA_p_stim_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/32.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice1/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_5/1x1_f_28_50_mA_p_stim_5_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/33.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice1/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_6/1x1_f_28_50_mA_p_stim_6_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/34.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice1/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_7/1x1_f_28_50_mA_p_stim_7_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/35.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice1/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_8/1x1_f_28_50_mA_p_stim_8_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/36.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice1/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_9/1x1_f_28_50_mA_p_stim_9_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/37.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice1/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_10/1x1_f_28_50_mA_p_stim_10_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/38.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice2/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_1/1x1_f_28_50_mA_p_stim_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/39.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice2/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_2/1x1_f_28_50_mA_p_stim_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/40.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice2/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_3/1x1_f_28_50_mA_p_stim_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/41.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice2/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_4/1x1_f_28_50_mA_p_stim_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/42.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice2/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_5/1x1_f_28_50_mA_p_stim_5_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/43.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice2/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_6/1x1_f_28_50_mA_p_stim_6_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/44.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice2/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_7/1x1_f_28_50_mA_p_stim_7_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/45.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice2/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_8/1x1_f_28_50_mA_p_stim_8_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/46.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice2/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_9/1x1_f_28_50_mA_p_stim_9_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/47.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice2/cell1/no_MLA/10_repeats/1x1_f_28_50_mA_p_stim_10/1x1_f_28_50_mA_p_stim_10_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/48.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice3/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_1/1x1_f_28_50_mA_p_stim_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/49.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice3/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_2/1x1_f_28_50_mA_p_stim_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/50.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice3/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_3/1x1_f_28_50_mA_p_stim_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/51.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice3/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_4/1x1_f_28_50_mA_p_stim_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/52.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice3/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_5/1x1_f_28_50_mA_p_stim_5_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/53.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice3/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_6/1x1_f_28_50_mA_p_stim_6_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/54.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice3/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_7/1x1_f_28_50_mA_p_stim_7_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/55.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice3/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_8/1x1_f_28_50_mA_p_stim_8_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/56.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice3/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_9/1x1_f_28_50_mA_p_stim_9_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/57.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice3/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_10/1x1_f_28_50_mA_p_stim_10_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/58.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice4/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_1/1x1_f_28_50_mA_p_stim_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/59.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice4/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_2/1x1_f_28_50_mA_p_stim_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/60.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice4/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_3/1x1_f_28_50_mA_p_stim_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/61.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice4/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_4/1x1_f_28_50_mA_p_stim_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/62.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice4/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_5/1x1_f_28_50_mA_p_stim_5_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/63.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice4/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_6/1x1_f_28_50_mA_p_stim_6_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/64.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice4/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_7/1x1_f_28_50_mA_p_stim_7_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/65.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice4/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_8/1x1_f_28_50_mA_p_stim_8_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/66.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice4/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_9/1x1_f_28_50_mA_p_stim_9_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/67.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice4/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_10/1x1_f_28_50_mA_p_stim_10_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/68.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice5/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_1/1x1_f_28_50_mA_p_stim_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/69.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice5/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_2/1x1_f_28_50_mA_p_stim_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/70.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice5/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_3/1x1_f_28_50_mA_p_stim_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/71.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice5/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_4/1x1_f_28_50_mA_p_stim_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/72.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice5/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_5/1x1_f_28_50_mA_p_stim_5_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/73.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice5/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_6/1x1_f_28_50_mA_p_stim_6_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/74.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice5/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_7/1x1_f_28_50_mA_p_stim_7_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/75.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice5/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_8/1x1_f_28_50_mA_p_stim_8_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/76.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice5/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_9/1x1_f_28_50_mA_p_stim_9_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/77.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice5/cell1/no_MLA/f_28_10_repeats/1x1_f_28_50_mA_p_stim_10/1x1_f_28_50_mA_p_stim_10_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/78.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice7/cell1/no_MLA/f_28_10_repeats/1x1_f_4_50_mA_p_stim_1/1x1_f_4_50_mA_p_stim_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/79.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice7/cell1/no_MLA/f_28_10_repeats/1x1_f_4_50_mA_p_stim_2/1x1_f_4_50_mA_p_stim_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/80.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice7/cell1/no_MLA/f_28_10_repeats/1x1_f_4_50_mA_p_stim_3/1x1_f_4_50_mA_p_stim_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/81.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice7/cell1/no_MLA/f_28_10_repeats/1x1_f_4_50_mA_p_stim_4/1x1_f_4_50_mA_p_stim_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/82.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice7/cell1/no_MLA/f_28_10_repeats/1x1_f_4_50_mA_p_stim_5/1x1_f_4_50_mA_p_stim_5_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/83.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice7/cell1/no_MLA/f_28_10_repeats/1x1_f_4_50_mA_p_stim_6/1x1_f_4_50_mA_p_stim_6_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/84.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice7/cell1/no_MLA/f_28_10_repeats/1x1_f_4_50_mA_p_stim_7/1x1_f_4_50_mA_p_stim_7_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/85.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice7/cell1/no_MLA/f_28_10_repeats/1x1_f_4_50_mA_p_stim_8/1x1_f_4_50_mA_p_stim_8_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/86.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice7/cell1/no_MLA/f_28_10_repeats/1x1_f_4_50_mA_p_stim_9/1x1_f_4_50_mA_p_stim_9_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/87.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice7/cell1/no_MLA/f_28_10_repeats/1x1_f_4_50_mA_p_stim_10/1x1_f_4_50_mA_p_stim_10_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/88.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice8/10_repeats/1x1_f_28_50_mA_p_stim_1/1x1_f_28_50_mA_p_stim_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/89.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice8/10_repeats/1x1_f_28_50_mA_p_stim_2/1x1_f_28_50_mA_p_stim_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/90.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice8/10_repeats/1x1_f_28_50_mA_p_stim_3/1x1_f_28_50_mA_p_stim_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/91.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice8/10_repeats/1x1_f_28_50_mA_p_stim_4/1x1_f_28_50_mA_p_stim_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/92.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice8/10_repeats/1x1_f_28_50_mA_p_stim_5/1x1_f_28_50_mA_p_stim_5_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/93.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice8/10_repeats/1x1_f_28_50_mA_p_stim_6/1x1_f_28_50_mA_p_stim_6_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/94.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice8/10_repeats/1x1_f_28_50_mA_p_stim_7/1x1_f_28_50_mA_p_stim_7_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/95.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice8/10_repeats/1x1_f_28_50_mA_p_stim_8/1x1_f_28_50_mA_p_stim_8_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/96.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice8/10_repeats/1x1_f_28_50_mA_p_stim_9/1x1_f_28_50_mA_p_stim_9_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/97.h5"))

    img_paths.append(os.path.join(in_dir, "/20181123/slice8/10_repeats/1x1_f_28_50_mA_p_stim_10/1x1_f_28_50_mA_p_stim_10_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/98.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice1/cell1/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_1/1x1_f_28_50_mA_p_stim_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/99.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice1/cell1/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_2/1x1_f_28_50_mA_p_stim_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/100.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice1/cell1/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_3/1x1_f_28_50_mA_p_stim_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/101.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice1/cell1/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_4/1x1_f_28_50_mA_p_stim_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/102.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice1/cell1/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_5/1x1_f_28_50_mA_p_stim_5_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/103.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice1/cell2/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_1/1x1_f_28_50_mA_p_stim_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/104.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice1/cell2/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_2/1x1_f_28_50_mA_p_stim_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/105.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice1/cell2/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_3/1x1_f_28_50_mA_p_stim_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/106.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice1/cell2/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_4/1x1_f_28_50_mA_p_stim_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/107.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice1/cell2/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_5/1x1_f_28_50_mA_p_stim_5_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/108.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice2/cell1/no_MLA/5_repeats_f_28/1x1_f_28_100_mA_p_stim_1/1x1_f_28_100_mA_p_stim_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/109.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice2/cell1/no_MLA/5_repeats_f_28/1x1_f_28_100_mA_p_stim_2/1x1_f_28_100_mA_p_stim_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/110.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice2/cell1/no_MLA/5_repeats_f_28/1x1_f_28_100_mA_p_stim_3/1x1_f_28_100_mA_p_stim_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/111.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice2/cell1/no_MLA/5_repeats_f_28/1x1_f_28_100_mA_p_stim_4/1x1_f_28_100_mA_p_stim_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/112.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice2/cell1/no_MLA/5_repeats_f_28/1x1_f_28_100_mA_p_stim_5/1x1_f_28_100_mA_p_stim_5_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/113.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f28/1x1_f_28_50_mA_p_stim_1/1x1_f_28_50_mA_p_stim_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/114.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f28/1x1_f_28_50_mA_p_stim_2/1x1_f_28_50_mA_p_stim_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/115.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f28/1x1_f_28_50_mA_p_stim_3/1x1_f_28_50_mA_p_stim_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/116.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f28/1x1_f_28_50_mA_p_stim_4/1x1_f_28_50_mA_p_stim_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/117.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f28/1x1_f_28_50_mA_p_stim_5/1x1_f_28_50_mA_p_stim_5_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/118.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f28/1x1_f_28_50_mA_p_stim_6/1x1_f_28_50_mA_p_stim_6_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/119.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f28/1x1_f_28_50_mA_p_stim_7/1x1_f_28_50_mA_p_stim_7_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/120.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f28/1x1_f_28_50_mA_p_stim_8/1x1_f_28_50_mA_p_stim_8_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/121.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f28/1x1_f_28_50_mA_p_stim_9/1x1_f_28_50_mA_p_stim_9_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/122.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f28/1x1_f_28_50_mA_p_stim_10/1x1_f_28_50_mA_p_stim_10_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/123.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f56_-10/1x1_f_28_50_mA_p_stim_1/1x1_f_28_50_mA_p_stim_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/124.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f56_-10/1x1_f_28_50_mA_p_stim_2/1x1_f_28_50_mA_p_stim_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/125.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f56_-10/1x1_f_28_50_mA_p_stim_3/1x1_f_28_50_mA_p_stim_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/126.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f56_-10/1x1_f_28_50_mA_p_stim_4/1x1_f_28_50_mA_p_stim_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/127.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f56_-10/1x1_f_28_50_mA_p_stim_5/1x1_f_28_50_mA_p_stim_5_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/128.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f56_-10/1x1_f_28_50_mA_p_stim_6/1x1_f_28_50_mA_p_stim_6_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/129.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f56_-10/1x1_f_28_50_mA_p_stim_7/1x1_f_28_50_mA_p_stim_7_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/130.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f56_-10/1x1_f_28_50_mA_p_stim_8/1x1_f_28_50_mA_p_stim_8_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/131.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f56_-10/1x1_f_28_50_mA_p_stim_9/1x1_f_28_50_mA_p_stim_9_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/132.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice3/cell1/no_MLA/10_repeats_f56_-10/1x1_f_28_50_mA_p_stim_10/1x1_f_28_50_mA_p_stim_10_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/133.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice4/cell1/no_MLA/10_repeats_f_28/1x1_f_28_50_mA_p_stim_1/1x1_f_28_50_mA_p_stim_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/134.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice4/cell1/no_MLA/10_repeats_f_28/1x1_f_28_50_mA_p_stim_2/1x1_f_28_50_mA_p_stim_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/135.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice4/cell1/no_MLA/10_repeats_f_28/1x1_f_28_50_mA_p_stim_3/1x1_f_28_50_mA_p_stim_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/136.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice4/cell1/no_MLA/10_repeats_f_28/1x1_f_28_50_mA_p_stim_4/1x1_f_28_50_mA_p_stim_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/137.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice4/cell1/no_MLA/10_repeats_f_28/1x1_f_28_50_mA_p_stim_5/1x1_f_28_50_mA_p_stim_5_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/138.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice4/cell1/no_MLA/10_repeats_f_28/1x1_f_28_50_mA_p_stim_6/1x1_f_28_50_mA_p_stim_6_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/139.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice4/cell1/no_MLA/10_repeats_f_28/1x1_f_28_50_mA_p_stim_7/1x1_f_28_50_mA_p_stim_7_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/140.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice4/cell1/no_MLA/10_repeats_f_28/1x1_f_28_50_mA_p_stim_8/1x1_f_28_50_mA_p_stim_8_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/141.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice4/cell1/no_MLA/10_repeats_f_28/1x1_f_28_50_mA_p_stim_9/1x1_f_28_50_mA_p_stim_9_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/142.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice4/cell1/no_MLA/10_repeats_f_28/1x1_f_28_50_mA_p_stim_10/1x1_f_28_50_mA_p_stim_10_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/143.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice6/cell1/no_MLA/10_f_28/1x1_f_28_50_mA_p_stim_1/1x1_f_28_50_mA_p_stim_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/144.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice6/cell1/no_MLA/10_f_28/1x1_f_28_50_mA_p_stim_2/1x1_f_28_50_mA_p_stim_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/145.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice6/cell1/no_MLA/10_f_28/1x1_f_28_50_mA_p_stim_3/1x1_f_28_50_mA_p_stim_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/146.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice6/cell1/no_MLA/10_f_28/1x1_f_28_50_mA_p_stim_4/1x1_f_28_50_mA_p_stim_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/147.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice6/cell1/no_MLA/10_f_28/1x1_f_28_50_mA_p_stim_5/1x1_f_28_50_mA_p_stim_5_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/148.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice6/cell1/no_MLA/10_f_28/1x1_f_28_50_mA_p_stim_6/1x1_f_28_50_mA_p_stim_6_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/149.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice6/cell1/no_MLA/10_f_28/1x1_f_28_50_mA_p_stim_7/1x1_f_28_50_mA_p_stim_7_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/150.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice6/cell1/no_MLA/10_f_28/1x1_f_28_50_mA_p_stim_8/1x1_f_28_50_mA_p_stim_8_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/151.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice6/cell1/no_MLA/10_f_28/1x1_f_28_50_mA_p_stim_9/1x1_f_28_50_mA_p_stim_9_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/152.h5"))

    img_paths.append(os.path.join(in_dir, "/20181122/slice6/cell1/no_MLA/10_f_28/1x1_f_28_50_mA_p_stim_10/1x1_f_28_50_mA_p_stim_10_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/153.h5"))

    img_paths.append(os.path.join(in_dir, "/20181121/slice5/cell2/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_1/1x1_f_28_50_mA_p_stim_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/154.h5"))

    img_paths.append(os.path.join(in_dir, "/20181121/slice5/cell2/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_2/1x1_f_28_50_mA_p_stim_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/155.h5"))

    img_paths.append(os.path.join(in_dir, "/20181121/slice5/cell2/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_3/1x1_f_28_50_mA_p_stim_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/156.h5"))

    img_paths.append(os.path.join(in_dir, "/20181121/slice5/cell2/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_4/1x1_f_28_50_mA_p_stim_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/157.h5"))

    img_paths.append(os.path.join(in_dir, "/20181121/slice5/cell2/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_5/1x1_f_28_50_mA_p_stim_5_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/158.h5"))

    img_paths.append(os.path.join(in_dir, "/20181121/slice6/cell1/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_1/1x1_f_28_50_mA_p_stim_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/159.h5"))

    img_paths.append(os.path.join(in_dir, "/20181121/slice6/cell1/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_2/1x1_f_28_50_mA_p_stim_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/160.h5"))

    img_paths.append(os.path.join(in_dir, "/20181121/slice6/cell1/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_3/1x1_f_28_50_mA_p_stim_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/161.h5"))

    img_paths.append(os.path.join(in_dir, "/20181121/slice6/cell1/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_4/1x1_f_28_50_mA_p_stim_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/162.h5"))

    img_paths.append(os.path.join(in_dir, "/20181121/slice6/cell1/no_MLA/5_repeats/1x1_f_28_50_mA_p_stim_5/1x1_f_28_50_mA_p_stim_5_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/163.h5"))

    img_paths.append(os.path.join(in_dir, "/20181112/slice3/cell1/no_MLA/repeats/1x1_p_stim_75_ma_1/1x1_p_stim_75_ma_1_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/164.h5"))

    img_paths.append(os.path.join(in_dir, "/20181112/slice3/cell1/no_MLA/repeats/1x1_p_stim_75_ma_2/1x1_p_stim_75_ma_2_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/165.h5"))

    img_paths.append(os.path.join(in_dir, "/20181112/slice3/cell1/no_MLA/repeats/1x1_p_stim_75_ma_3/1x1_p_stim_75_ma_3_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/166.h5"))

    img_paths.append(os.path.join(in_dir, "/20181112/slice3/cell1/no_MLA/repeats/1x1_p_stim_75_ma_4/1x1_p_stim_75_ma_4_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/167.h5"))

    img_paths.append(os.path.join(in_dir, "/20181112/slice3/cell1/no_MLA/repeats/1x1_p_stim_75_ma_5/1x1_p_stim_75_ma_5_MMStack_Pos0.ome.tif"))
    out_paths.append(os.path.join(out_dir, "/168.h5"))


    for i in range(len(img_paths)):
        process(img_paths[i], out_paths[i])