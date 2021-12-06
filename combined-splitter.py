import PIL
from pathlib import Path
from PIL import UnidentifiedImageError
from PIL import Image
#from osgeo import gdal
import rasterio as rs

#path = Path("../test/PlanetNewImagery/test/planetImageryForestsGreater2Hect").rglob("*.tif")
#out_path = "../planetSplit/test/planetImageryForestsGreater2Hect/" 
path = Path("../data/combined/test/combinedPlanetForests").rglob("*.tif")
out_path = "../planet/test/combinedPlanetForests/"

for img_p in path:
    #if str(img_p[36:-4]) < 'way308294349':
    #    if img_p[36] != 'r':
    #        continue
    startindex = len(path) + 1
    m1img_out_path = out_path +  str(img_p)[startindex:-4]+'m1'+str(img_p)[-4:]

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
    profile.update(dtype=rs.uint16, count=3)
    
    with rs.open(m1img_out_path, 'w', **profile) as dst:
        dst.write(m1.astype(rs.uint16))

print('finished test data forests')

#path = Path("../test/PlanetNewImagery/train/planetImageryForestsGreater2Hect").rglob("*.tif")
#out_path = "../planetSplit/train/planetImageryForestsGreater2Hect/" 
path = Path("../data/combined/train/combinedPlanetForests").rglob("*.tif")
out_path = "../planet/train/combinedPlanetForests/"
for img_p in path:
    #if str(img_p[36:-4]) < 'way308294349':
    #    if img_p[36] != 'r':
    #        continue
    startindex = len(path) + 1
    m1img_out_path = out_path +  str(img_p)[startindex:-4]+'m1'+str(img_p)[-4:]

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
    profile.update(dtype=rs.uint16, count=3)
    
    with rs.open(m1img_out_path, 'w', **profile) as dst:
        dst.write(m1.astype(rs.uint16))

print('finished train data forests')

path = Path("../data/combined/val/combinedPlanetForests").rglob("*.tif")
out_path = "../planet/val/combinedPlanetForests/" 
for img_p in path:
    #if str(img_p[36:-4]) < 'way308294349':
    #    if img_p[36] != 'r':
    #        continue
    startindex = len(path) + 1
    m1img_out_path = out_path +  str(img_p)[startindex:-4]+'m1'+str(img_p)[-4:]

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
    profile.update(dtype=rs.uint16, count=3)
    
    with rs.open(m1img_out_path, 'w', **profile) as dst:
        dst.write(m1.astype(rs.uint16))

print('finished val data forests')
print('starting ORCHARDS data')
path = Path("../data/combined/test/combinedPlanetOrchards").rglob("*.tif")
out_path = "../planet/test/combinedPlanetOrchards/"

for img_p in path:
    #if str(img_p[36:-4]) < 'way308294349':
    #    if img_p[36] != 'r':
    #        continue
    startindex = len(path) + 1
    m1img_out_path = out_path +  str(img_p)[startindex:-4]+'m1'+str(img_p)[-4:]


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
    profile.update(dtype=rs.uint16, count=3)
    
    with rs.open(m1img_out_path, 'w', **profile) as dst:
        dst.write(m1.astype(rs.uint16))

print('finished test data orchards')

#path = Path("../test/PlanetNewImagery/train/planetImageryForestsGreater2Hect").rglob("*.tif")
#out_path = "../planetSplit/train/planetImageryForestsGreater2Hect/" 
path = Path("../data/combined/train/combinedPlanetOrchards").rglob("*.tif")
out_path = "../planet/train/combinedPlanetOrchards/"
for img_p in path:
    #if str(img_p[36:-4]) < 'way308294349':
    #    if img_p[36] != 'r':
    #        continue
    startindex = len(path) + 1
    m1img_out_path = out_path +  str(img_p)[startindex:-4]+'m1'+str(img_p)[-4:]

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
    profile.update(dtype=rs.uint16, count=3)
    
    with rs.open(m1img_out_path, 'w', **profile) as dst:
        dst.write(m1.astype(rs.uint16))

print('finished train data orchards')

path = Path("../data/combined/val/combinedPlanetOrchards").rglob("*.tif")
out_path = "../planet/val/combinedPlanetOrchards/" 
for img_p in path:
    #if str(img_p[36:-4]) < 'way308294349':
    #    if img_p[36] != 'r':
    #        continue
    startindex = len(path) + 1
    m1img_out_path = out_path +  str(img_p)[startindex:-4]+'m1'+str(img_p)[-4:]

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
    profile.update(dtype=rs.uint16, count=3)
    
    with rs.open(m1img_out_path, 'w', **profile) as dst:
        dst.write(m1.astype(rs.uint16))
