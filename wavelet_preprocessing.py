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

def dec_single_level_gray_combined(img, percent):
    LL, (LH, HL, HH) = pywt.dwt2(img, "db8", mode="periodization")
    p = np.percentile(abs(LH), percent)
    LH[abs(LH) < p] = 0 # zero coefficients close to 0
    p = np.percentile(abs(HL), percent)
    HL[abs(HL) < p] = 0
    p = np.percentile(abs(HH), percent)
    HH[abs(HH) < p] = 0
    combined = LH + HL + HH # adding three high freq images together
    return combined

def dec_single_level_gray(img, percent):
    c = pywt.dwt2(img, "db8", mode="periodization")
    c[0] = np.zeros_like(c[0]) # throw away thumbnail
    arr, slices = pywt.coeffs_to_array(c)
    p = np.percentile(abs(arr), percent)
    arr[abs(arr) < p] = 0 # zero coefficients only close to 0
    ad = arr[slices[1]['ad']]
    dd = arr[slices[1]['dd']]
    da = arr[slices[1]['da']]
    stacked = np.stack([ad, dd, da])
    return stacked

def dec_img_multichannel(img, l, percent):
    decs = []
    for channel in img:
        decs.append(dec_img(channel, l, percent))

    stacked = np.vstack(decs)
    return stacked # returns shape [2*num_channels, H/2^l, W/2^l]