"""
McDonald lab miniscope data processing pipeline
Heat maps, place cells detection, Bayesian decoding and all the good stuff :)

Copyright (C) 2023 - HaoRan Chang
SPDX-License-Identifier: GPL-2.0-or-later
"""
import os
import csv
import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from fast_histogram import histogram2d
from scipy import stats
from scipy import signal
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from .behav import moving

def read_ts(file):
    with open(file) as f:
        reader = csv.reader(f)
        data = []
        for entry in reader:
            data.append(entry)
    
    data = np.vstack(data[1:])
    ts = data[:, 1]
    return ts

def sync_ts(beh_ts, scope_ts):
    searcher = BallTree(beh_ts.reshape(-1, 1), metric='l2')
    sync_idx = searcher.query(scope_ts.reshape(-1, 1), k=1, return_distance=False).flatten()
    searcher = BallTree(scope_ts.reshape(-1, 1), metric='l2')
    unsync_idx = searcher.query(beh_ts.reshape(-1, 1), k=1, return_distance=False).flatten()

    return sync_idx, unsync_idx


def mk_bins(pos, nbins=50):
    dims = np.ptp(pos, axis=1)
    bins = [np.round(nbins*dims[0]/dims[1]), nbins] if dims[0] > dims[1] else [nbins, np.round(nbins*dims[1]/dims[0])]
    bins = np.array(bins).astype(int)
    range = np.array([np.min(pos, axis=1), np.max(pos, axis=1)]).T
    return bins, range

def mk_hmap(pos, deconv, smooth=False, nbins=50, sigma=3, range=None, bins=None):
    if bins is None:
        bins, _ = mk_bins(pos, nbins)
    if range is None:
        _, range = mk_bins(pos, nbins)

    occ = histogram2d(*pos, bins=bins, range=range)
    occ = occ[:, :, np.newaxis]
    hmap = [histogram2d(*pos[:, w>0], range=range, bins=bins, weights=w[w>0]) for w in deconv] # sparse compute
    hmap = np.stack(hmap, axis=2)
    hmap = np.divide(hmap, occ, out=np.zeros_like(hmap), where=occ!=0)
    occ = occ / np.sum(occ)
    r, c, _ = np.nonzero(occ == 0)
    if smooth:
        V = gaussian_filter(hmap, sigma=sigma, axes=(0, 1))
        W = gaussian_filter((occ>0).astype(float), sigma=sigma)
        smooth_map = np.divide(V, W, out=np.zeros_like(V), where=W!=0)
        smooth_map[r, c, :] = np.nan
    else:
        smooth_map = None
    hmap[r, c, :] = np.nan
    return hmap, occ, smooth_map

def SI(hmap, occ, deconv):
    I = np.divide(hmap, np.mean(deconv, axis=1), out=np.zeros_like(hmap), where=~np.isnan(hmap))
    I = np.sum(occ * I * np.log2(I, out=np.zeros_like(I), where=np.logical_and(I!=0, ~np.isnan(I))), axis=(0, 1))
    return I

def SI_test(pos, deconv, nperms=1000):
    print('running spatial information permutation test...')
    null = np.empty((deconv.shape[0], nperms))
    hmap, occ, _ = mk_hmap(pos, deconv)
    si = SI(hmap, occ, deconv)
    for i in range(nperms):
        shift = np.random.randint(deconv.shape[1])
        p = np.roll(pos, shift, axis=1)
        scramp, _, _ = mk_hmap(p, deconv)
        null[:, i] = SI(scramp, occ, deconv)
        print("%2.2f %%" % (i/nperms*100), end='\r')
    p = np.sum(si[:, np.newaxis] <= null, axis=1) / nperms
    return p

def split_test(pos, deconv, nbins=50, nperms=200):
    print('running split half spatial correlation test...')
    bins, ranger = mk_bins(pos, nbins)
    
    disc = np.empty_like(pos)
    for i in range(len(bins)):
        disc[i] = np.digitize(pos[i], bins=np.linspace(*ranger[i], bins[i]+1))
    
    disc[disc == (nbins+1)] = nbins
    _, indices, count = np.unique(disc, axis=1, return_inverse=True, return_counts=True)
    
    split = np.full_like(indices, False, dtype=bool)
    for i in range(len(count)):
        idx = np.where(indices == i)[0]
        idx = idx[:np.ceil(count[i] / 2).astype(int)]
        split[idx] = True
    
    left, _, _ = mk_hmap(pos[:, split], deconv[:, split], range=ranger, bins=bins, smooth=True)
    right, _, _ = mk_hmap(pos[:, ~split], deconv[:, ~split], range=ranger, bins=bins, smooth=True)
    
    def get_R(x, y):
        R = stats.pearsonr(x, y)
        return R.statistic
    
    R = np.empty(left.shape[2])
    R_pval = np.empty(left.shape[2])
    for i in range(left.shape[2]):
        l = left[:, :, i].flatten()
        r = right[:, :, i].flatten()
        idx = np.logical_and(~np.isnan(l), ~np.isnan(r))
        res = stats.permutation_test((l[idx], r[idx]), get_R, permutation_type='pairings',\
                                     n_resamples=nperms, alternative='greater')
        R[i] = res.statistic
        R_pval[i] = res.pvalue
        print("%2.2f %%" % (i/left.shape[2]*100), end='\r')

    return R_pval, R

def mape(pos, deconv, k=10, tau=30, nbins=50, sigma=5, normalize=True, penalty=2**-52, rm_idle=True):
    # Maximum a posteriori estimation
    # params:
    #     k:          k-fold cross-validation
    #     tau:        window length
    #     nbins:      number of bins for heat maps and decoding
    #     sigma:      heat map smoothing factor
    #     normalize:  normalize dF/F by STD, usually leads to better results
    #     penalty:    replace log(0) = -inf with a constant penalty
    #     rm_idle:    reject non-moving epochs

    def decode(pos, deconv, d, tau, ranger, bins, sigma, penalty):
        _, occ, hmap = mk_hmap(pos, deconv, smooth=True, range=ranger, bins=bins, sigma=sigma)
        with np.errstate(divide='ignore'):
            l_hmap = np.log(hmap)
            l_occ = np.log(occ).squeeze()
        l_hmap[l_hmap == -np.inf] = penalty
        sigma_f = np.sum(hmap, axis=2).squeeze() * tau

        mape = np.empty((2, d.shape[3]))
        t = time.time()
        for i in range(d.shape[3]):
            pmap = np.sum(l_hmap * d[:, :, :, i], axis=2) - sigma_f + l_occ
            pmap[np.isnan(pmap)] = -np.inf
            mape[:, i] = np.unravel_index(np.argmax(pmap), pmap.shape)
            print("%2.2f %%\tETA: %3.2f sec" % (i/d.shape[3]*100, (time.time() - t)/(i+1)*(d.shape[3]-i-1)), end='\r')
        
        return mape

    
    sum_win = np.ones(tau).T
    if rm_idle:
        mvt = moving(pos, axis=1)
    else:
        mvt = np.ones(pos.shape[1]) > 0
    
    tmp = np.repeat(np.arange(k), np.ceil(np.sum(mvt) / k))
    tmp = tmp[:np.sum(mvt)]
    cv = np.zeros(pos.shape[1])
    cv[mvt] = tmp
    
    if normalize:
        deconv = deconv / np.expand_dims(np.std(deconv, axis=1), axis=1)
    
    d = signal.fftconvolve(deconv, sum_win[np.newaxis, :], mode='same', axes=1)
    d = d[:, :, np.newaxis, np.newaxis].transpose(2, 3, 0, 1)
    
    bins, ranger = mk_bins(pos, nbins)
    
    mape = np.empty_like(pos)
    t = time.time()
    for c in range(k):
        print("running CV %2d/%2d:" % (c+1, k))
        train = c != cv
        train[~mvt] = False
        test = c == cv
        test[~mvt] = False
        mape[:, test] = decode(pos[:, train], deconv[:, train], d[:, :, :, test], tau, ranger, bins, sigma, penalty)
        #print("%2.2f %%\tETA: %3.2f sec" % (c/(k-1)*100, (time.time() - t)/(c+1)*(k-c)), end='\r')

    print("running idle epochs")
    mape[:, ~mvt] = decode(pos[:, mvt], deconv[:, mvt], d[:, :, :, ~mvt], tau, ranger, bins, sigma, penalty)
        
    decoded = mape * np.diff(ranger, axis=1) / bins[:, np.newaxis] + ranger[:, 0, np.newaxis]
    #decoded = gaussian_filter1d(decoded, sigma=tau/2)
    err = np.sqrt(np.sum((decoded - pos)**2, axis=0))
    print("mean decoding error: %4.3f" %(np.mean(err[mvt])))

    return decoded, err