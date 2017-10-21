from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
import numpy as np

# methods from https://www.kaggle.com/c/data-science-bowl-2017#tutorial

def do_thresholding(img):
    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img/std

    middle = img[100:400,100:400] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    #move the underflow bins
    img[img==max]=mean
    img[img==min]=mean
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    return thresh_img

def do_lungmask(img):

    # Thresholding
    thresh_img = do_thresholding(img)

    # do erosion and dilation
    eroded = morphology.erosion(thresh_img,np.ones([4,4]))
    dilation = morphology.dilation(eroded,np.ones([10,10]))
    labels = measure.label(dilation)
    label_vals = np.unique(labels)

    # cut non-ROI regions
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
            good_labels.append(prop.label)
    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0
    #
    #  The mask here is the mask for the lungs--not the nodes
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    return mask

def do_final_processing(img, mask):
    # apply lung mask
    img = mask * img   
    #
    # renormalizing the masked image (in the mask region)
    #
    new_mean = np.mean(img[mask>0])  
    new_std = np.std(img[mask>0])
    #
    #  Pulling the background color up to the lower end
    #  of the pixel range for the lungs
    #
    old_min = np.min(img)       # background color
    img[img==old_min] = new_mean-1.2*new_std   # resetting backgound color
    img = img-new_mean
    img = img/new_std
    #make image bounding box  (min row, min col, max row, max col)
    labels = measure.label(mask)
    regions = measure.regionprops(labels)
    #
    # Finding the global min and max row over all regions
    #
    min_row = 512
    max_row = 0
    min_col = 512
    max_col = 0
    for prop in regions:
        B = prop.bbox
        if min_row > B[0]:
            min_row = B[0]
        if min_col > B[1]:
            min_col = B[1]
        if max_row < B[2]:
            max_row = B[2]
        if max_col < B[3]:
            max_col = B[3]
    width = max_col-min_col
    height = max_row - min_row
    if width > height:
        max_row=min_row+width
    else:
        max_col = min_col+height
    # 
    # cropping the image down to the bounding box for all regions
    # (there's probably an skimage command that can do this in one line)
    # 
    img = img[min_row:max_row,min_col:max_col]
    mask =  mask[min_row:max_row,min_col:max_col]
    new_img = img
    if max_row-min_row <5 or max_col-min_col<5:  # skipping all images with no god regions
        return None
    else:
        # moving range to -1 to 1 to accomodate the resize function
        mean = np.mean(img)
        img = img - mean
        min = np.min(img)
        max = np.max(img)
        img = img/(max-min)
        new_img = resize(img,[512,512])
    return new_img
