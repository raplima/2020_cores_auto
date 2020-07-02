# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:24:07 2020

@author: Rafael
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

def plot_res (cfg, samp, im, outputs, patch_dict, thing_classes):
    """
    function to plot results

    Parameters
    ----------
    cfg : detectron2 config file

    samp : dictionary
        one sample from the (detectron2) Dataset
    im : numpy array
        the image (read by cv2) from samp
    outputs : dict
        a dictionary with detectron2.structures.instances.Instances, 
        the prediction of im.
    patch_dict : dictionary
        a dictionary containing color (perhaps patch) info for the classes.
    thing_classes : list
        a list of classes

    Returns
    -------
    fig : matplotlib figure
        the figure built by this function

    """
    im = cv2.imread(samp['file_name'])

    fig, ax = plt.subplots(ncols=3, figsize=(10,7))
    ax[0].imshow(im)
    ax[1].imshow(im)
    ax[2].imshow(im)

    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()

    ax[0].set_title('Input image')
    ax[1].set_title('Interpretation')
    ax[2].set_title('Prediction')

    # add letters:
    for ii, l in enumerate('ABC'):
        ax[ii].text(0, 0,
                    s=l, 
                    bbox=dict(facecolor='white', alpha=1), 
                    ha='left', va='bottom', size=14)
        
    # add colors:
    jj = 200
    for k in patch_dict:
        ax[2].text(im.shape[1], jj,
                    s=k, 
                    bbox=dict(facecolor=patch_dict[k]['color'], alpha=0.7), 
                    ha='left', va='bottom', size=14)
        jj += 500


    for annot in samp['annotations']:
        # select the x and y segmentation points
        segx = annot['segmentation'][0][0::2]
        segy = annot['segmentation'][0][1::2]
        ax[1].fill(segx, segy, patch_dict[thing_classes[annot['category_id']]]['color'], alpha=0.8)


    for ii in range(len(outputs['instances'])):
        # select the x and y segmentation points
        if outputs['instances'].scores[ii].item() > cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
            mask = np.array(outputs['instances'].pred_masks[ii].tolist())
            contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            c = np.array([[v[0][0], v[0][1]] for v in contours[0]])

            segx=c[:,0]
            segy=c[:,1]
            ax[2].fill(segx, segy, 
                           patch_dict[thing_classes[outputs['instances'].pred_classes[ii].item()]]['color'], 
                           alpha=0.8)
    return fig