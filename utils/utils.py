'''
Description: 
2022-04-13 21:07:37
Created by Kai Zhou.
Email address is kz4yolo@gmail.com.
'''
import pickle
from skimage import morphology
from skimage import measure
from skimage.filters import median
from scipy import ndimage
import numpy as np
import glob

def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)

def get_u(labels, list_seg, tolerance=[30,30]):
    abs_center = [labels.shape[1]//2, labels.shape[2]//2]
    res_label = list()
    for u,c in list_seg[1:5]:
        coord = np.where(labels == u)
        for center_x,center_y in zip(coord[2],coord[1]):
            if np.abs(center_y - abs_center[0]) < tolerance[0] and np.abs(center_x - abs_center[1]) < tolerance[1]:
                res_label.append(u)
                break 
    return res_label

def binarize_lung_mask(dcm_array, threshold, smoothing=True):
    '''Get the lung mask from DICOM Array.
    Parameters
    ----------
    dcm_array : DICOM Array.
    threshold : a list of threholding value, 
            like [A, B], A is the lower bound and B is the upper bound.
    smoothing : bool value, 
            If True, the lung_mask will be smoothed by median (skimage.filters).
    
    Returns
    -----------
    binarize_lung_mask : ndarray
            the binarized lung mask
            
    '''
    threshold_mask_lower = dcm_array >= threshold[0]
    threshold_mask_upper = dcm_array <= threshold[1]
    threshold_mask = np.multiply(threshold_mask_lower, threshold_mask_upper)
    labels = measure.label(threshold_mask,connectivity=1)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg = list(zip(unique, counts))[1:]
    list_seg = sorted(list_seg,key=lambda x:x[1], reverse=True) # sorted
    u = get_u(labels, list_seg)
    lung_mask = np.zeros_like(dcm_array)
    for u_i in u: 
        lung_mask[np.where(labels == u_i)] = 1 
    if smoothing:
        lung_mask = median(lung_mask, behavior='ndimage')
    return lung_mask.astype(np.ubyte)

def segment_lung_mask(ct_path):
#     ct_array, ct_img = read_mha(ct_path)
    ct_array = np.load(ct_path)
    lung_array = binarize_lung_mask(ct_array,[-1024,-500],False)
#     airway_array, _ = read_mha(ct_path.replace(ct_path.split("/")[-1], "airway.mha"))
#     lung_array = lung_array - airway_array
    res_lung = np.zeros_like(lung_array)
    for i in range(lung_array.shape[0]):
        lung_slice = lung_array[i]
        eroded = morphology.binary_erosion(lung_slice,np.ones([4,4]))  
        dilation = morphology.binary_dilation(eroded,np.ones([8,8]))  
        labels = measure.label(dilation)   
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels) # 获取连通区域
        
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
                good_labels.append(prop.label)
        '''
        (0L, 0L, 512L, 512L)
        (190L, 253L, 409L, 384L)
        (200L, 110L, 404L, 235L)
        '''
        # 根据肺部标签获取肺部mask，并再次进行’膨胀‘操作，以填满并扩张肺部区域
        mask = np.ndarray(lung_slice.shape,dtype=np.int8)
        mask[:] = 0
        for N in good_labels:
            mask = mask + np.where(labels==N,1,0)
        mask = morphology.binary_dilation(mask,np.ones([5,5])) # one last dilation
        mask = ndimage.binary_fill_holes(mask,np.ones([5,5]))
        mask = morphology.binary_erosion(mask,np.ones([5,5]))
        mask = morphology.binary_erosion(mask,np.ones([3,3]))
        res_lung[i] = mask
#     np.save(res_lung.astype(np.ubyte), target_path)
    return res_lung.astype(np.ubyte)

def get_best_ckpt(cp_dir):
    cp_list = glob.glob(os.path.join(cp_dir,"*.pth"))
    best_model = cp_list[0]
    best_cp = int(cp_list[0].split("/")[-1].split("_")[2])
    for cp in cp_list[1:]:
        ep_idx = int(cp.split("/")[-1].split("_")[2])
        if ep_idx > best_cp:
            best_cp = ep_idx
            best_model = cp
    return best_model

def resample(img, size, order=1):
    from scipy.ndimage.interpolation import zoom
    from scipy.ndimage import zoom
    resize_factor = np.array(size)/np.array(img.shape)
    #img = zoom(img, resize_factor, mode='nearest', order=order)
    img = zoom(img, resize_factor, order=1)
    return img

def lung_crop(ct, lung_mask, margin=5):
    assert ct.shape == lung_mask.shape, "Wrong shape. ct shape:{}, lung_mask shape:{}".format(ct.shape, lung_mask.shape)
    
    coord = np.where(lung_mask >0)
    coord_z_min = coord[0].min() if len(coord[0])>0 else 0
    coord_z_max = coord[0].max() if len(coord[0])>0 else ct.shape[0]
    z_s = max(coord_z_min-margin, 0)
    z_e = min(coord_z_max+margin, ct.shape[0])
    
    coord_y_min = coord[1].min() if len(coord[1])>0 else 0
    coord_y_max = coord[1].max() if len(coord[1])>0 else ct.shape[1]
    y_s = max(coord_y_min-margin, 0)
    y_e = min(coord_y_max+margin, ct.shape[1])
    
    coord_x_min = coord[2].min() if len(coord[2])>0 else 0
    coord_x_max = coord[2].max() if len(coord[2])>0 else ct.shape[2]
    x_s = max(coord_x_min-margin, 0)
    x_e = min(coord_x_max+margin, ct.shape[2])
    
    crop_ct = ct[z_s:z_e, y_s:y_e, x_s:x_e]
    return crop_ct
