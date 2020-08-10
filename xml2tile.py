import sys, os, glob
import math
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import openslide
import staintools

from preprocess import apply_image_filters, tissue_percent

def get_downsampled_image(slide, scale_factor=32):
    large_w, large_h=slide.dimensions
    new_w = math.floor(large_w/scale_factor)
    new_h = math.floor(large_h/scale_factor)    
    level = slide.get_best_level_for_downsample(scale_factor)
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), Image.BILINEAR)
    return img, new_w, new_h

def get_start_end_coordinates(x, tile_size):
    start = int(x * tile_size)
    end = int((x+1) * tile_size)
    return start, end

def get_stain_normalizer(path='/path/to/reference/image', method='vahadane'):
    target = staintools.read_image(path)
    target = staintools.LuminosityStandardizer.standardize(target)
    normalizer = staintools.StainNormalizer(method=method)
    normalizer.fit(target)
    return normalizer

def apply_stain_norm(tile, normalizer):
    to_transform = np.array(tile).astype('uint8')
    to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
    transformed = normalizer.transform(to_transform)
    transformed = Image.fromarray(transformed)
    return transformed

if __name__ == '__main__':
    xml_rootpath='/path/contains/xml_files/'
    slide_rootpath='/path/contains/whole_slide_images/'
    out_rootpath='/path/to/save/image/tiles/'
    scale_factor=32
    tile_size=1024
    normalizer = get_stain_normalizer()

    tile_names = []
    labels = []

    xmlfiles=os.listdir(xml_rootpath)
    for i in range(len(xmlfiles)):

        xmlfile = xmlfiles[i]
        savepath = out_rootpath + xmlfile[:-4]+'/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        slide_path=slide_rootpath+xmlfile[:-4]+'.svs'
        slide = openslide.open_slide(slide_path)
        
        tree = ET.parse(xml_rootpath+xmlfile)
        root = tree.getroot()
        verts=[]
        for vertices in root.findall("./Annotation/Regions/Region/Vertices"):
            v=[]
            for vertex in vertices.iter('Vertex'):
                a = int(vertex.attrib['X'])
                b = int(vertex.attrib['Y'])
                v.append([a,b])
            verts.append(v)
        verts_list = [np.array(i) for i in verts]
        mask = np.zeros((int(slide.dimensions[1]),int(slide.dimensions[0])), dtype=np.uint8)
        cv2.fillPoly(mask, verts_list, 1)
        
        dim_w, dim_h = int(slide.level_dimensions[-1][0]/10), int(slide.level_dimensions[-1][1]/10)
        small_mask = cv2.resize(mask, (dim_w, dim_h), cv2.INTER_AREA)
        small_mask_img = Image.fromarray(np.uint8(small_mask*255))
        small_img = slide.get_thumbnail((dim_w, dim_h))
        mask_alpha = small_mask_img.convert('RGBA')
        img_alpha = small_img.convert('RGBA').resize(small_mask_img.size)
        alphaBlended = Image.blend(img_alpha, mask_alpha, alpha=.4)
        alphaBlended.save(out_rootpath+xmlfile[:-4]+'_mask_overlay.png')

        prop = slide.properties
        if prop['aperio.AppMag']=='40':

            img, new_w, new_h = get_downsampled_image(slide, scale_factor=scale_factor)
            tissue=apply_image_filters(np.array(img))

            small_tile_size = int(((tile_size/scale_factor)*2+1)//2)
            num_tiles_h = new_h//small_tile_size
            num_tiles_w = new_w//small_tile_size

            for h in range(num_tiles_h):
                for w in range(num_tiles_w):
                    small_start_h, small_end_h = get_start_end_coordinates(h, small_tile_size)
                    small_start_w, small_end_w = get_start_end_coordinates(w, small_tile_size)
                    tile_region = tissue[small_start_h:small_end_h, small_start_w:small_end_w]
                    if tissue_percent(tile_region)>=80:
                        try:
                                start_h, end_h = get_start_end_coordinates(h, tile_size)
                                start_w, end_w = get_start_end_coordinates(w, tile_size)
                                tile_path = savepath+slide_list[i]+'_'+str(tile_size)+'_x'+str(start_w)+'_'+str(w)+'_'+str(num_tiles_w)+'_y'+str(start_h)+'_'+str(h)+'_'+str(num_tiles_h)+'.png'
                                
                                if os.path.exists(tile_path):
                                    print('%s Alraedy Tiled' % (tile_path))
                                else:
                                    tile = slide.read_region((start_w, start_h), 0, (tile_size, tile_size))
                                    tile = tile.convert("RGB")
                                    transformed = apply_stain_norm(tile, normalizer)
                                    transformed.save(tile_path)
                        
                                    tile_mask=mask[start_h:end_h, start_w:end_w]
                                    # label: 0=non-tumor, 1=border, 2=tumor
                                    if tile_mask.sum()==0:
                                        label=0
                                    elif (tile_mask==0).any()==False:
                                        label=2
                                    else:
                                        label=1
                                    labels.append(label)
                                    tile_name = slide_list[i]+'_'+str(tile_size)+'_x'+str(start_w)+'_'+str(w)+'_'+str(num_tiles_w)+'_y'+str(start_h)+'_'+str(h)+'_'+str(num_tiles_h)+'.png'
                                    tile_names.append(tile_name)
                        except:
                            print('error for %s' % (tile_path))  
            print('Done for %s' % xmlfile[:-4])
        else:
            print('AppMag is %s for %s' % (prop['aperio.AppMag'], slide_list[i]))

    df = pd.DataFrame(columns=['tile_name', 'label'])
    df.tile_name = tile_names
    df.label = labels
    df.to_csv('/path/to/save/csv', index=False)
    


