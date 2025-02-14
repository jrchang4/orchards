import PIL
from pathlib import Path
from PIL import UnidentifiedImageError
from PIL import Image
#from osgeo import gdal
import rasterio as rs

#path = Path("../test/PlanetNewImagery/test/planetImageryForestsGreater2Hect").rglob("*.tif")
#out_path = "../planetSplit/test/planetImageryForestsGreater2Hect/" 
path = Path("../test/data2/val/planetImageryCentroidOrchards").rglob("*.tif")
out_path = "../planetSplit/val/planetImageryOrchardsGreater2Hect/"

for img_p in path:
    #if str(img_p[36:-4]) < 'way308294349':
    #    if img_p[36] != 'r':
    #        continue
    m1img_out_path = out_path +  str(img_p)[50:-4]+'m1'+str(img_p)[-4:]

    '''
    m2img_out_path = out_path +  str(img_p)[31:-4]+'m2'+str(img_p)[-4:]
    m3img_out_path = out_path +  str(img_p)[31:-4]+'m3'+str(img_p)[-4:]
    m4img_out_path = out_path +  str(img_p)[31:-4]+'m4'+str(img_p)[-4:]
    m5img_out_path = out_path +  str(img_p)[31:-4]+'m5'+str(img_p)[-4:]
    m6img_out_path = out_path +  str(img_p)[31:-4]+'m6'+str(img_p)[-4:]
    m7img_out_path = out_path +  str(img_p)[31:-4]+'m7'+str(img_p)[-4:]
    '''
    src = rs.open(img_p)
    array = src.read()
    m1 = array[0:3]
    m2 = array[4:8]
    m3 = array[8:12]
    m4 = array[12:16]
    m5 = array[16:20]
    m6 = array[20:24]
    m7 = array[24:28]
    
    
    profile = src.profile
    profile.update(dtype=rs.uint8, count=3)
    
    with rs.open(m1img_out_path, 'w', **profile) as dst:
        dst.write(m1.astype(rs.uint8))

print('finished test data orchards')

#path = Path("../test/PlanetNewImagery/train/planetImageryForestsGreater2Hect").rglob("*.tif")
#out_path = "../planetSplit/train/planetImageryForestsGreater2Hect/" 
path = Path("../test/data2/train/planetImageryCentroidOrchards").rglob("*.tif")
out_path = "../planetSplit/train/planetImageryOrchardsGreater2Hect/"
for img_p in path:
    #if str(img_p[36:-4]) < 'way308294349':
    #    if img_p[36] != 'r':
    #        continue
    m1img_out_path = out_path +  str(img_p)[50:-4]+'m1'+str(img_p)[-4:]

    '''
    m2img_out_path = out_path +  str(img_p)[31:-4]+'m2'+str(img_p)[-4:]
    m3img_out_path = out_path +  str(img_p)[31:-4]+'m3'+str(img_p)[-4:]
    m4img_out_path = out_path +  str(img_p)[31:-4]+'m4'+str(img_p)[-4:]
    m5img_out_path = out_path +  str(img_p)[31:-4]+'m5'+str(img_p)[-4:]
    m6img_out_path = out_path +  str(img_p)[31:-4]+'m6'+str(img_p)[-4:]
    m7img_out_path = out_path +  str(img_p)[31:-4]+'m7'+str(img_p)[-4:]
    '''
    src = rs.open(img_p)
    array = src.read()
    m1 = array[0:3]
    m2 = array[4:8]
    m3 = array[8:12]
    m4 = array[12:16]
    m5 = array[16:20]
    m6 = array[20:24]
    m7 = array[24:28]
    
    
    profile = src.profile
    profile.update(dtype=rs.uint8, count=3)
    
    with rs.open(m1img_out_path, 'w', **profile) as dst:
        dst.write(m1.astype(rs.uint8))

print('finished train data orchards')


