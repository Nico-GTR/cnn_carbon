import ee
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

def initialize_ee():
    """
    Initialize Google Earth Engine API.
    Attempts to initialize with a specific project ID from environment variables.
    Falls back to authentication flow if initialization fails.
    """
    project_id = os.environ.get('EE_PROJECT_ID')
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()

def get_stratified_points(region, pool_size):
    """
    Generate balanced coordinates: 50% Forest / 50% Non-forest.
    
    Args:
        region (ee.Geometry): Target geographical area.
        pool_size (int): Large buffer of points to account for rejections.
        
    Returns:
        tuple: (list of coordinates, ESA AGB image object)
    """
    agb_dataset = ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB") \
                    .filterDate('2020-01-01', '2020-12-31') \
                    .first() \
                    .select('AGB')

    binary_forest = agb_dataset.gt(50).rename('forest_class')

    print(f"Generating a reservoir of {pool_size} stratified coordinates...")
    
    points = binary_forest.stratifiedSample(
        numPoints=pool_size // 2,
        classBand='forest_class',
        region=region,
        scale=100,
        geometries=True
    )
    
    return points.geometry().coordinates().getInfo(), agb_dataset

def extract_patches(target_patches=500):
    """
    Main extraction pipeline. Iterates through a large coordinate pool 
    until EXACTLY 'target_patches' valid samples are downloaded.
    """
    initialize_ee()
    
    region_oise = ee.Geometry.Rectangle([2.0, 49.0, 3.5, 49.8])
    
    # Over-sampling: Request 3x more points to act as a buffer
    pool_size = target_patches * 3 
    coords, agb_img = get_stratified_points(region_oise, pool_size)
    
    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
               .filterDate('2020-01-01', '2020-12-31') \
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
               .median()
    
    x_data = []
    y_data = []

    # Initialize manual progress bar tracking VALID patches, not total attempts
    with tqdm(total=target_patches, desc=f"Extracting {target_patches} valid patches") as pbar:
        for lon, lat in coords:
            
            # Stop condition: We reached the exact required number of patches
            if len(x_data) >= target_patches:
                break
                
            point = ee.Geometry.Point([lon, lat])
            roi = point.buffer(1280).bounds()
            
            try:
                patch_s2 = s2_col.reproject(crs='EPSG:3857', scale=10) \
                                 .sampleRectangle(region=roi, defaultValue=0) \
                                 .getInfo()
                
                label = agb_img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=roi,
                    scale=100
                ).getInfo().get('AGB')

                if label is not None:
                    b4 = np.array(patch_s2['properties']['B4'])
                    b3 = np.array(patch_s2['properties']['B3'])
                    b2 = np.array(patch_s2['properties']['B2'])
                    b8 = np.array(patch_s2['properties']['B8'])
                    
                    img = np.dstack((b4, b3, b2, b8))
                    h, w, c = img.shape
                    
                    if h >= 256 and w >= 256:
                        start_y = int(h / 2) - 128
                        start_x = int(w / 2) - 128
                        
                        img_cropped = img[start_y:start_y+256, start_x:start_x+256, :]
                        
                        x_data.append(img_cropped)
                        y_data.append(label)
                        
                        # Increment progress bar ONLY on success
                        pbar.update(1)
                        
            except Exception:
                continue

    # Persistence
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "X_train.npy", np.array(x_data, dtype=np.float32))
    np.save(output_dir / "y_train.npy", np.array(y_data, dtype=np.float32))
    print(f"\nExtraction complete: Exactly {len(x_data)} patches saved to {output_dir}")

if __name__ == "__main__":
    extract_patches(500)