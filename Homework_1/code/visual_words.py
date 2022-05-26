import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import sklearn.cluster
import multiprocessing
from functools import partial
import os

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)

    [hint]
    * To produce the expected collage, loop first over scales, then filter types, then color channel.
    * Note the order argument when using scipy.ndimage.gaussian_filter. 
    '''
    
    filter_scales = opts.filter_scales
    # ----- TODO -----
    if len(img.shape) == 2:     # if the image is a gray-scale
        img = np.repeat(img[..., np.newaxis], 3, 2)
    img = skimage.color.rgb2lab(img)
    output = np.empty(img.shape, dtype=float)
    for scale in filter_scales:
        # (1) Gaussian (2) Laplacian of Gaussian (3) derivative of Gaussian in the x direction (4) derivative of Gaussian in the y direction
        for type in range(1, 5):
            im_filtered = np.zeros(img.shape)
            if type == 2:
                for channel in range(3):
                    im_filtered[:, :, channel] = scipy.ndimage.gaussian_laplace(img[:, :, channel], sigma=scale)
            else:
                type_order = {1: 0, 3: (0, 1), 4: (1, 0)}
                for channel in range(3):
                    im_filtered[:, :, channel] = scipy.ndimage.gaussian_filter(img[:, :, channel], sigma=scale, order=type_order[type])
            output = np.append(output, im_filtered, axis=2)
    return output[:, :, 1:]   # output[:,:,0] is zeros

def compute_dictionary_one_image(img_path, opts):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    img_path = join(opts.data_dir, img_path)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    responses = extract_filter_responses(opts, img)
    # reshape (M,N,3F) to (MxN,3F)
    responses_reshape = responses.reshape(responses.shape[0] * responses.shape[1], responses.shape[2])
    np.random.shuffle(responses_reshape)
    filter_output = responses_reshape[0:opts.alpha, :]      # select alpha pixels randomly
    save_path = 'tmp_' + str(img_path.split('/')[-1][:-4])
    np.save(save_path, filter_output)
    pass

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    #  For testing purpose, you can create a train_files_small.txt to only load a few images.
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----
    p = multiprocessing.Pool(processes=n_worker)
    partial_compute = partial(compute_dictionary_one_image, opts=opts)
    p.map(partial_compute, train_files)

    for i in range(len(train_files)):
        # e.g. 'aquarium/sun_asgtepdmsxsrqqvy.jpg' —— 'tmp_sun_asgtepdmsxsrqqvy.npy'
        cur_path = 'tmp_' + str(train_files[i].split('/')[-1][:-4]) + '.npy'
        cur = np.load(cur_path)
        os.remove(cur_path)
        if i == 0:
            filter_responses = np.array(cur)
            continue
        filter_responses = np.concatenate((filter_responses, cur), axis=0)

    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    np.save(join(out_dir, 'dictionary.npy'), dictionary)
    pass

    ## example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    responses = extract_filter_responses(opts, img)
    # reshape (M,N,3F) to (MxN,3F)
    responses_reshape = responses.reshape(responses.shape[0] * responses.shape[1], responses.shape[2])
    # e.g. row[0]:[distance(pixel_0, cluster center 0),distance(pixel_0, cluster center 1)...]
    dis = scipy.spatial.distance.cdist(responses_reshape, dictionary)   # (MxN,K)
    min_rows = np.argmin(dis, axis=1)    # the mini_dis between pixels and centers (MxN,)
    wordmap = min_rows.reshape(img.shape[0], img.shape[1])
    return wordmap