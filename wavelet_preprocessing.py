import numpy as np
import pywt
from matplotlib import pyplot as plt

def dec_img(img, l, percent):
    c = pywt.wavedec2(img, "db8", mode='periodization', level=l)
    arr, slices = pywt.coeffs_to_array(c)
    p = np.percentile(abs(arr), percent)
    arr[abs(arr) < p] = 0 # zero coefficients only close to 0
    thumbnail = arr[slices[0]] # get slices of image
    ad = arr[slices[1]['ad']]
    dd = arr[slices[1]['dd']]
    da = arr[slices[1]['da']]
    combined = ad + dd + da # adding three high freq images together
    stacked = np.stack([thumbnail, combined], axis=0) # returns shape [2, H/2^l, W/2^l]
    return stacked

def dec_img_multichannel(img, l, percent):
    decs = []
    for channel in img:
        decs.append(dec_img(channel, l, percent))

    stacked = np.vstack(decs)
    return stacked # returns shape [2*num_channels, H/2^l, W/2^l]