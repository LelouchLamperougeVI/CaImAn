"""
McDonald lab behaviour videos extraction module

Copyright (C) 2023 - HaoRan Chang
SPDX-License-Identifier: GPL-2.0-or-later
"""
import os
import re
import cv2
import numpy as np
from multiprocessing import Pool
import time
import h5py

def __mkbg(target, nFrames=20): # make background for subtraction
    cap = cv2.VideoCapture(target)
    idx = (cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1) * np.linspace(0, 1, nFrames)
    
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        frames.append(f)
    
    cap.release()
    bg = np.median([f for f in frames], axis=0).astype('uint8')
    return bg

def moving(pos, wdw=30, quantile=.25, axis=0): # detect movement epochs
    vel = np.diff(pos, axis=axis)
    vel = np.sqrt(np.sum(vel**2, axis=axis-1))
    
    # RMS envelope
    if wdw > pos.shape[axis]:
        wdw = pos.shape[axis] - 1
    wdw = np.ones(wdw) / wdw
    rms = np.sqrt(np.convolve(vel**2, wdw, 'same'))
    mvt = rms > np.quantile(rms[rms>0], quantile)
    mvt = np.append(mvt, mvt[-1])

    return mvt

def unflip(pos, heading, sigma=6): # make sure heading vector points in the right direction
    circdist = lambda theta: np.min([np.abs(theta), 2 * np.pi - np.abs(theta)], axis=0) * np.sign(theta)
    circnorm = lambda theta: np.array([x - 2*np.pi if x > np.pi else x for x in theta % (2*np.pi)]).flatten()

    theta = heading
    delta = np.diff(theta, axis=0);
    delta = circdist(delta)
    
    flip_points = np.abs(delta) > np.std(delta) * sigma #six sigma baby!
    flip_points[[-1, 0]] = True
    
    vel = np.diff(pos, axis=0)
    mtheta = np.arctan2(vel[:, 1], vel[:, 0])
    mvt = moving(pos)
    mtheta[~mvt[:-1]] = np.nan
    mtheta = mtheta.flatten()
    
    idx = np.transpose(flip_points.nonzero())
    idx[0] = -1
    idx[-1] -= 1
    
    heading = theta
    for i in zip(idx[:-1, 0] + 1, idx[1:, 0]):
        e1 = np.nansum( np.abs(circdist(theta[i[0]:i[1]] - mtheta[i[0]:i[1]])) )
        e2 = circnorm(theta[i[0]:i[1]] - np.pi)
        e2 = np.nansum( np.abs(circdist(e2 - mtheta[i[0]:i[1]])) ) # this is L1 norm
        if e2 < e1:
            heading[i[0]:i[1]] = circnorm(heading[i[0]:i[1]] - np.pi)
            
    return pos, heading
    
def player(target, show=False, pos=None, heading=None, marker=None): # for both playing and processing behaviour videos
    bg = __mkbg(target)
    cap = cv2.VideoCapture(target)

    if pos is None:
        x = []
    if heading is None:
        theta = []
    flipper = 1
    buff = np.nan
    count = 0
    while(cap.isOpened()):
      ret, f = cap.read()
      if ret == True:
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        g = cv2.absdiff(g, bg)
        _, t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(t, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        idx = areas.index(max(areas))
    
        M = cv2.moments(contours[idx])
        if pos is None:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x.append([cx, cy])
        else:
            cx = pos[count, 0]
            cy = pos[count, 1]

        if heading is None:
            rows, cols = t.shape
            vx, vy, _, _ = cv2.fitLine(contours[idx], cv2.DIST_L2,0,0.01,0.01)
              
            if np.abs(vy - buff) > 1:
                flipper = -flipper
            buff = vy
            vx = vx * flipper
            vy = vy * flipper
            theta.append([vx, vy])
        else:
            vx = np.cos(heading[count])
            vy = np.sin(heading[count])

        if show:
            cv2.arrowedLine(f,(int(cx), int(cy)),(int(cx + vx * 100), int(cy + vy * 100)),(0,255,0),2)
            cv2.drawContours(f, contours, idx, (0, 0, 255), 2)
            if marker is not None:
                cv2.drawMarker(f, (int(marker[count, 0]), int(marker[count, 1])), color=(255, 255, 0), markerType=cv2.MARKER_STAR, thickness=2)
            cv2.imshow('Frame', f)
            cv2.setWindowTitle('Frame', str(cap.get(cv2.CAP_PROP_POS_FRAMES)))
         
            if cv2.waitKey(25) & 0xFF == ord('q'):
              break

        count += 1
          
      else: 
        break
    
    cap.release()
    cv2.destroyAllWindows()
    if pos is None:
        pos = np.array(x)
    if heading is None:
        theta = np.array(theta)
        heading = np.arctan2(theta[:, 1], theta[:, 0]).flatten()

    pos, heading = unflip(pos, heading)
    return pos, heading

class processor():
    def __init__(self, path, verbose=False):
        self.__root = path
        files = [f for f in os.listdir(self.__root) if re.match(r'\d+.avi', f)]
        order = [int(re.search(r'(\d+).avi', f).group(1)) for f in files]
        files = [x for _, x in sorted(zip(order, files))]
        self.files = [os.path.join(self.__root, x) for x in files]
        self.processed = np.full(len(self.files), False)
        self.pos = [[] for _ in self.files]
        self.heading = [[] for _ in self.files]

    def get(self, idx=None):
        if idx is None:
            idx = [i for (i, item) in enumerate(self.processed) if item]
        pos = np.vstack([self.pos[i] for i in idx])
        heading = [self.heading[i] for i in idx]
        files = [self.files[i] for i in idx]
        idx = np.repeat(idx, [len(x) for x in heading])
        heading = np.hstack(heading)

        return pos, heading, files, idx

    def save(self):
        pos, heading, files, idx = self.get()
        
        path = os.path.join(self.__root, 'behaviour.h5')
        hf = h5py.File(path, 'w')
        hf.create_dataset('position', data=pos)
        hf.create_dataset('heading', data=heading)
        hf.create_dataset('files', data=files)
        hf.create_dataset('indices', data=idx)
        hf.close()
        print("behaviour data exported as\t%s" % path)

    def list(self):
        print("[ind][processed]\t filename")
        for (i, item) in enumerate(self.files):
            processed = 'X' if self.processed[i] else ' '
            print("[%3d][%s]\t%s" % (i, processed, item))
    
    def runall(self):
        t = time.time()
        print("processing all remaining behaviour videos...")
        idx = [i for (i, item) in enumerate(self.processed) if ~item]
        process_pool = Pool(os.cpu_count())
        for i, result in enumerate(process_pool.imap_unordered(self.run, idx)):
            self.pos[result[2]] = result[0]
            self.heading[result[2]] = result[1]
            self.processed[result[2]] = True
            print("(%3d/%3d)\t%s\tETA:%5.3f sec" % (i+1, len(idx), self.files[result[2]], (len(idx)-i-1)*(time.time()-t)/(i+1)))
        print('done!')
        print("execution time: %5.3f seconds." % (time.time() - t))
        print('unflitting all heading vectors...')
        pos, heading, _, idx = self.get()
        pos, heading = unflip(pos, heading)
        self.pos = [pos[i == idx, :] for i in np.unique(idx)]
        self.heading = [heading[i == idx] for i in np.unique(idx)]

    def run(self, target):
        pos, heading = player(self.files[target])

        self.pos[target] = pos
        self.heading[target] = heading
        self.processed[target] = True
        return pos, heading, target

    def show(self, target):
        if not self.processed[target]:
            raise RuntimeError("Target video not yet processed. Run behav.processor.run(target) first or runall.")
        player(self.files[target], show=True, pos=self.pos[target], heading=self.heading[target])
        