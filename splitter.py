import PIL
from pathlib import Path
from PIL import UnidentifiedImageError
from PIL import Image
#from osgeo import gdal
import rasterio as rs

path = Path("../test/planetImageryCentroidForests").rglob("*.tif")
out_path = "../planetSinglesForests" 
for img_p in path:
    m1img_out_path = out_path +  str(img_p)[36:-4]+'m1'+str(img_p)[-4:]

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
    '''
    with rs.open(m2img_out_path, 'w', **profile) as dst:
        dst.write(m2.astype(rs.uint8),4)
    with rs.open(m3img_out_path, 'w', **profile) as dst:
        dst.write(m3.astype(rs.uint8),4)
    with rs.open(m4img_out_path, 'w', **profile) as dst:
        dst.write(m4.astype(rs.uint8),4)
    with rs.open(m5img_out_path, 'w', **profile) as dst:
        dst.write(m5.astype(rs.uint8),4)
    with rs.open(m6img_out_path, 'w', **profile) as dst:
        dst.write(m6.astype(rs.uint8),4)
    with rs.open(m7img_out_path, 'w', **profile) as dst:
        dst.write(m7.astype(rs.uint8),4)
    
    '''

'''
path = Path("../test/planetImageryCentroidOrchards").rglob("*.tif")
out_path = "../planetSinglesOrchards" 
for img_p in path:
    m1img_out_path = out_path +  str(img_p)[37:-4]+'m1'+str(img_p)[-4:]
    m2img_out_path = out_path +  str(img_p)[31:-4]+'m2'+str(img_p)[-4:]
    m3img_out_path = out_path +  str(img_p)[31:-4]+'m3'+str(img_p)[-4:]
    m4img_out_path = out_path +  str(img_p)[31:-4]+'m4'+str(img_p)[-4:]
    m5img_out_path = out_path +  str(img_p)[31:-4]+'m5'+str(img_p)[-4:]
    m6img_out_path = out_path +  str(img_p)[31:-4]+'m6'+str(img_p)[-4:]
    m7img_out_path = out_path +  str(img_p)[31:-4]+'m7'+str(img_p)[-4:]
    
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
    profile.update(dtype=rs.uint8,count=3)
    
    with rs.open(m1img_out_path, 'w', **profile) as dst:
        dst.write(m1.astype(rs.uint8))

    
    with rs.open(m2img_out_path, 'w', **profile) as dst:
        dst.write(m2.astype(rs.uint8),4)
    with rs.open(m3img_out_path, 'w', **profile) as dst:
        dst.write(m3.astype(rs.uint8),4)
    with rs.open(m4img_out_path, 'w', **profile) as dst:
        dst.write(m4.astype(rs.uint8),4)
    with rs.open(m5img_out_path, 'w', **profile) as dst:
        dst.write(m5.astype(rs.uint8),4)
    with rs.open(m6img_out_path, 'w', **profile) as dst:
        dst.write(m6.astype(rs.uint8),4)
    with rs.open(m7img_out_path, 'w', **profile) as dst:
        dst.write(m7.astype(rs.uint8),4)

    

'''
