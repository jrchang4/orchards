import os
import numpy as np
import random
from PIL import Image
from osgeo import gdal
from matplotlib import pyplot as plt


def calculate_rgb_simple_stats(path):
    res_means = []
    res_vars = []
    for i, filename in enumerate(os.listdir(path)):
        if i % 200 == 0:
            print(i)
        with Image.open(os.path.join(path, filename)) as im:
            np_img = np.asarray(im)
        res_means.append(np.mean(np_img, axis=(0,1)))
        res_vars.append(np.std(np_img, axis=(0,1)))
    var = np.mean(np.array(res_vars), axis=0)
    means = np.mean(np.array(res_means), axis=0)
    return var,means

DATA_PATH = '../planetSplit/val' #replace ctoups22 with your own user / data path
FORESTS_PATH = os.path.join(DATA_PATH, "ImagesGoogleMapsForests")
ORCHARDS_PATH = os.path.join(DATA_PATH, "ImagesGoogleMapsOrchards")
FORESTS_TIF_PATH = os.path.join(DATA_PATH, "planetImageryForestsGreater2Hect")
ORCHARDS_TIF_PATH = os.path.join(DATA_PATH, "planetImageryOrchardsGreater2Hect")
random.seed(25)



orchards_stats = calculate_rgb_simple_stats(ORCHARDS_PATH)
forests_stats = calculate_rgb_simple_stats(FORESTS_PATH)

orchard_vars = orchards_stats[0]
orchards_means = orchards_stats[1]
forest_vars = forests_stats[0]
forest_means = forests_stats[1]

def calibrate_image(rgb_image, ref_stds, ref_means):
    # Transform test image to 32-bit floats to avoid 
    # surprises when doing arithmetic with it 
    calibrated_img = rgb_image.copy().astype('float32')

    # Loop over RGB
    for i in range(3):
        # Subtract mean 
        calibrated_img[:,:,i] = calibrated_img[:,:,i]-np.nanmean(calibrated_img[:,:,i])
        # Normalize variance
        calibrated_img[:,:,i] = calibrated_img[:,:,i]/np.nanstd(calibrated_img[:,:,i])
        # Scale to reference 
        calibrated_img[:,:,i] = calibrated_img[:,:,i]*ref_stds[i] + ref_means[i]
        # Clip any values going out of the valid range
        calibrated_img[:,:,i] = np.clip(calibrated_img[:,:,i],0,255)

    # Convert to 8-bit unsigned int
    return calibrated_img.astype('uint8')

# 'i' is the starting index. If you want the first months RGB, then pass i = 1, for example. 
# month two would start at i=5
def get_rgb_from_tif(tif_dataset_object, i):
    band = tif_dataset_object.GetRasterBand(i)
    blue = band.ReadAsArray()
    
    band = tif_dataset_object.GetRasterBand(i+1)
    green = band.ReadAsArray()
    
    band = tif_dataset_object.GetRasterBand(i+2)
    red = band.ReadAsArray()
    
    rgb = np.stack((red, green, blue), axis=2)
    return rgb
def correct_image(full_path, i, columns=None, len_images=None):
    gdal_dataset = gdal.Open(full_path)
    rgb = get_rgb_from_tif(gdal_dataset, i)
    
    #plt.figure()
    
    test_image_calibrated = calibrate_image(rgb, forest_vars, forest_means)
#    if columns:
#        plt.subplot(len_images / columns + 1, columns, i + 1)
    plt.imshow(test_image_calibrated)

    #rgb = get_rgb_from_tif(gdal_dataset, i=1)
    
    #test_image_calibrated = calibrate_image(rgb, forest_vars, forest_means)
    #if columns:
    #    plt.subplot(len_images / columns + 1, columns, i + 1)
    #plt.imshow(test_image_calibrated)

    '''
    rgb = get_rgb_from_tif(gdal_dataset, i=5)
    
    test_image_calibrated = calibrate_image(rgb, forest_vars, forest_means)
    if columns:
        plt.subplot(len_images / columns + 1, columns, i + 1)
    plt.imshow(test_image_calibrated)

    rgb = get_rgb_from_tif(gdal_dataset, i=9)
    
    test_image_calibrated = calibrate_image(rgb, forest_vars, forest_means)
    if columns:
        plt.subplot(len_images / columns + 1, columns, i + 1)
    plt.imshow(test_image_calibrated)

    rgb = get_rgb_from_tif(gdal_dataset, i=13)
    
    test_image_calibrated = calibrate_image(rgb, forest_vars, forest_means)
    if columns:
        plt.subplot(len_images / columns + 1, columns, i + 1)
    plt.imshow(test_image_calibrated)

    rgb = get_rgb_from_tif(gdal_dataset, i=17)
    
    test_image_calibrated = calibrate_image(rgb, forest_vars, forest_means)
    if columns:
        plt.subplot(len_images / columns + 1, columns, i + 1)
    plt.imshow(test_image_calibrated)

    rgb = get_rgb_from_tif(gdal_dataset, i=21)
    
    test_image_calibrated = calibrate_image(rgb, forest_vars, forest_means)
    if columns:
        plt.subplot(len_images / columns + 1, columns, i + 1)
    plt.imshow(test_image_calibrated)

    rgb = get_rgb_from_tif(gdal_dataset, i=5)
    
    test_image_calibrated = calibrate_image(rgb, forest_vars, forest_means)
    if columns:
        plt.subplot(len_images / columns + 1, columns, i + 1)
    plt.imshow(test_image_calibrated)
    '''




# +
#good ones = [way659611919.tif, ]
def generate_corrected_images(directory_path, i):
    dir_tif_images = os.listdir(directory_path)
    #pathslist = ['']
    #test_image_paths = [(dir_tif_images + x) for x in pathslist]#random.sample(dir_tif_images, 5)
#    test_image_paths = random.sample(dir_tif_images,1)
#    test_image_paths = ['way931938597.tif']

#for orchards: 
    test_image_paths = ['way208404091m1.tif, way297299914m1.tif, way298129942m1.tif, way311662751m1.tif, way311993958m1.tif, way313495549m1.tif, way467268998
m1.tif, way497159037m1.tif, way660247155m1.tif']
    print(test_image_paths)
    plt.figure(figsize=(10,10))
    columns = 3
    
#    for i, path in enumerate(test_image_paths):
    for path in test_image_paths:
        correct_image(os.path.join(directory_path, path), i, columns, len(test_image_paths))
#         gdal_dataset = gdal.Open(os.path.join(directory_path, path))
#         rgb = get_rgb_from_tif(gdal_dataset, i=1)
#         test_image_calibrated = calibrate_image(rgb, forest_vars, forest_means)
#         plt.subplot(len(test_image_paths) / columns + 1, columns, i + 1)
#         plt.imshow(test_image_calibrated)
# -



generate_corrected_images(FORESTS_TIF_PATH,1)

#generate_corrected_images(FORESTS_TIF_PATH,5)

#generate_corrected_images(FORESTS_TIF_PATH,9)

#generate_corrected_images(FORESTS_TIF_PATH,13)

#generate_corrected_images(FORESTS_TIF_PATH,17)

#generate_corrected_images(FORESTS_TIF_PATH,21)

#generate_corrected_images(FORESTS_TIF_PATH,25)

generate_corrected_images(ORCHARDS_TIF_PATH,1)

#generate_corrected_images(ORCHARDS_TIF_PATH,5)

#generate_corrected_images(ORCHARDS_TIF_PATH,9)
