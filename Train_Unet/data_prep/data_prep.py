import SimpleITK as sitk
import os
import numpy as np
from PIL import Image


class counts():
    '''
    a counts class that determines how many unique classes exist and what their labels
    '''
    def __init__(self):
        self.arrs = []

    def count_unique(self, arr):
        '''
        a method that counts the unique values in the array
        :param arr: the label to count
        :return: 0
        '''
        self.arrs.append(np.unique(arr))

    def count(self):
        '''
        a method that counts the unique classes from the sublist of unique classes
        :return:0
        '''
        self.uni = np.unique([u for sublist in self.arrs for u in sublist])


def count_classes(src, path='ct_train'):
    '''
    a method that counts the number of classes and stores them. This is for re-indexing and to know the number of classes
    without hardcoding
    :param src: source directory
    :param path: train or test path
    :return: 0
    '''
    cc = counts()
    dest = os.path.join(src, path)
    for file in os.listdir(dest):
        full_p = os.path.join(dest, file)
        im = sitk.ReadImage(full_p)
        im_arr = sitk.GetArrayFromImage(im)
        name = file.split(sep='_')
        desig = name[3].split(sep='.')[0] + "_"
        if desig == 'label_':
            cc.count_unique(im_arr)
    cc.count()
    np.save(os.path.join(os.path.split(dest)[0], 'class_counts'), cc.uni)
    return cc.uni

def re_index(arr,classes):
    '''
    a function that reindexes the labels. It transforms label a into label a', label b into label b' etc
    :param arr: input array to reindex
    :param classes: class values to reindex
    :return: array with re-indexed classes
    '''
    new_idx = np.arange(len(classes))
    for i,cls in enumerate(classes):
        arr[arr == cls] = new_idx[i]
    return arr


def resize_img(arr,img_size):
    '''
    a function that resizes the image to a smaller size to allow for faster optimization
    :param arr: array to resize
    :param img_size: new size
    :return: np array with new size
    '''
    im = Image.fromarray(arr)
    if im.mode == "I;16":
        im = im.convert('P')
    return np.array(im.resize((img_size,img_size),Image.NEAREST))



def load_and_save(src, paths, norm, img_size=512):
    '''
    a function that checks if .npy files exist, and if not creates them from the DICOM files
    :param src: source data path
    :param paths: train and test paths
    :return: image/label names, number of classes
    '''
    try:
        classes = np.load(os.path.join(src,'class_counts.npy'))
    except:
        print("Counting number of classes now")
        classes = count_classes(src)
    r_im = []
    r_lbl = []
    for p in paths:
        names_im = []
        names_lbl = []
        dest = os.path.join(src,p)
        if not any(os.path.splitext(f)[1] == '.npy' for f in os.listdir(dest)):
            print("No npy files detected in {}. Creating them now".format(p))
            for file in os.listdir(dest):
                header = dict()
                full_p = os.path.join(dest,file)
                im = sitk.ReadImage(full_p)
                keys = im.GetMetaDataKeys()
                im_arr = sitk.GetArrayFromImage(im)
                for k in keys:
                    header[k] = im.GetMetaData(k)
                name = file.split(sep='_')
                desig = name[3].split(sep='.')[0] + "_"
                if desig == 'image_':
                    #normalzie the image
                    im_arr = normalize(im_arr, norm)
                else:
                    #reindex the label
                    im_arr = re_index(im_arr, classes)
                for i in range(im_arr.shape[0]):
                    im = im_arr[i]
                    #split images into 2D slices
                    if desig == 'image_':
                        names_im.append(os.path.join(dest,desig+name[2]+"_slice_" + str(i)+".npy"))
                    else:
                        names_lbl.append(os.path.join(dest,desig+name[2]+"_slice_" + str(i)+".npy"))
                    if img_size != 512:
                        im = resize_img(im, img_size)
                    np.save(os.path.join(dest, desig + name[2] + "_slice_" + str(i)), im)
                np.save(os.path.join(dest,desig+"header_"+name[2]), np.array(header))
            print("Done creating npy files in ",p)
        else:
            #generates names even if preprocessing was already done, for dataloading
            names_im = [f for f in os.listdir(dest) if os.path.splitext(f)[1] == '.npy' and f.split('_')[0] == 'image' and f.split('_')[1] != 'header']
            names_lbl = [f for f in os.listdir(dest) if os.path.splitext(f)[1] == '.npy' and f.split('_')[0] == 'label'and f.split('_')[1] != 'header']

        r_im.append(names_im)
        r_lbl.append(names_lbl)
    return (r_im, r_lbl), classes

def normalize(im, norm='per_patient'):
    '''
    a function that normalizes an array
    :param im: impate to normalize
    :param norm: how to normalize
    :return: normalzied array
    '''
    if norm == 'per_patient':
        return (im - im.min())/(im.max() - im.min())
    else:
        raise NotImplementedError
