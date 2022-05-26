import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image
import matplotlib
import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    wordmap = wordmap.flatten()
    hist = np.bincount(wordmap, minlength=K)
    total = sum(hist)
    hist = hist/total
    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----
    hist_all = np.empty((0,), dtype=float)
    H, W = wordmap.shape
    weights = np.array([0.25]*(L+1))
    for i in range(2, L+1):
        weights[i] = weights[i-1]*2

    # calculate the last layer(layer=L) and store hist results in memo
    step_row, step_col = H // (2 ** L), W // (2 ** L)
    last_hist = np.zeros((2**L, 2 ** L, K))
    for i in range(2**L):
        for j in range(2**L):
            cur_cell = wordmap[i*step_row:(i+1)*step_row, j*step_col:(j+1)*step_col]    # extract cur cell
            last_hist[i, j, :] = get_feature_from_wordmap(opts, cur_cell)   # store the hist of cur cell
    hist_all = np.append(last_hist.flatten() * weights[L], hist_all)

    # calculate layers(<L) using the results in the previous layer
    pre_hist = np.copy(last_hist)
    for layer in range(L - 1, -1, -1):
        cur_hist = np.zeros((2 ** layer, 2 ** layer, K))
        for i in range(2 ** layer):
            for j in range(2 ** layer):
                cur_hist[i, j, :] = np.sum(pre_hist[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2, :], axis=(0, 1))
        hist_all = np.append(cur_hist.flatten() * weights[layer], hist_all)
        pre_hist = cur_hist

    hist_all = hist_all/sum(hist_all)
    return hist_all
    
def get_image_feature(args):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    # ----- TODO -----
    opts, img_path, dictionary = args
    img_path = join(opts.data_dir, img_path)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)
    save_path = 'tmp_feature' + str(img_path.split('/')[-1][:-4])
    np.save(save_path, feature)

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----
    p = multiprocessing.Pool(processes=n_worker)
    list_args = [(opts, img_path, dictionary) for img_path in train_files]
    p.map(get_image_feature, list_args)
    for i in range(len(train_files)):
        # e.g. 'aquarium/sun_asgtepdmsxsrqqvy.jpg' —— 'tmp_feature_sun_asgtepdmsxsrqqvy.npy'
        cur_path = 'tmp_feature' + str(train_files[i].split('/')[-1][:-4]) + '.npy'
        cur = np.load(cur_path)
        os.remove(cur_path)
        if i == 0:
            features = np.array(cur)
            continue
        features = np.vstack((features, cur))

    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

def similarity_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    mins = np.minimum(word_hist, histograms)
    return np.sum(mins, axis=1)
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    trained_features = trained_system['features']
    trained_labels = trained_system['labels']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----
    p = multiprocessing.Pool(processes=n_worker)
    list_args = [(opts, img_path, dictionary) for img_path in test_files]
    p.map(get_image_feature, list_args)
    predict_labels = np.zeros(len(test_labels))
    for i in range(len(test_files)):
        # e.g. 'aquarium/sun_asgtepdmsxsrqqvy.jpg' —— 'tmp_feature_sun_asgtepdmsxsrqqvy.npy'
        cur_path = 'tmp_feature' + str(test_files[i].split('/')[-1][:-4]) + '.npy'
        cur = np.load(cur_path)
        os.remove(cur_path)
        sim = similarity_to_set(cur, trained_features)
        predict_labels[i] = trained_labels[np.argmax(sim)]  # find the label with the largest sim

    conf = np.zeros((8, 8), dtype=int)
    for i in range(len(predict_labels)):
        conf[int(test_labels[i]), int(predict_labels[i])] += 1

    accuracy = np.trace(conf)/np.sum(conf)
    return conf, accuracy
