import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import misc
from scipy import signal as sg
import random
import array
import scipy.stats as st
import timeit
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool




def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


class Dude:

    ind = 1
    mutationRate = 10 # parts per 1000, ie, 1 = 0.1%

    def __init__(self, WH, kSize):
        """ Initializes a dude for the population with a complete set of Genome for both layers of the Neural Net """
        self.WH = WH
        self.kSize = kSize
        self.gen1 = np.random.random((WH*kSize, WH*kSize))
        self.gen1 = self.gen1 * 2 -1
        self.gen2 = np.random.random((WH*kSize, WH*kSize))
        self.gen2 = self.gen1 * 2 -1
        self.normalize(self.gen1)
        self.normalize(self.gen2)
        self.mutations = 0
        self.name = "DUDE-" + str(Dude.ind)
        Dude.ind = Dude.ind + 1

    def die(self):
        return

    def normalize(self, mat):
        k = self.kSize
        for y in range(0,self.WH):
            for x in range(0, self.WH):
                mat[x*k:(x+1)*k, y*k:(y+1)*k] = self.normalizeSlice(mat[x*k:(x+1)*k, y*k:(y+1)*k])


    def normalizeSlice(self, mat):
        sum = mat.sum()
        if sum == 0:
            return mat
        else:
            return mat/sum

    def process(self, input, test):
        level1 = self.processGenomeSet(self.gen1, input)
        self.output = self.processGenomeSet(self.gen2, level1)
        self.fitness = self.getFitness(self.output, test)
        return self.output

    def processAsync(self, input, test):
        level1 = self.processGenomeSet1(self.gen1, input)
        self.output = self.processGenomeSet1(self.gen2, level1)
        self.fitness = self.getFitness(self.output, test)
        return self.output

    def getFitness(self, output, test):
        return abs(output-test).sum()/((self.WH*self.kSize)**2)

    def getBounds(self, v, k, s):
        if 0 > v - k//2:
            return [0, k]
        if s <= v + k//2:
            return [s-k,s]
        return [v-k//2, v + k//2 + 1]

    def processGenomeSet1(self, genomeSet, input):
        levelResult = np.empty(input.shape)
        k = self.kSize
        s = len(input[0])
        xr = [0,0]
        yr = [0,0]
        y = 0
        while y < s:
            x = 0

            while x < s:
                xr = self.getBounds(x, k, s)
                yr = self.getBounds(y, k, s)
                scalar = self.convArea(input[xr[0]:xr[1], yr[0]:yr[1]], genomeSet[x*k:(x+1)*k,y*k:(y+1)*k])
                levelResult[x,y] = scalar
                x += 1
            y += 1

        return levelResult

    def processGenomeSetAsync(self, genomeSet, input):
        k = self.kSize
        s = len(input[0])

        def cellIter(v, idx):
            x, y = idx
            xr = self.getBounds(x, k, s)
            yr = self.getBounds(y, k, s)
            return self.convArea(input[xr[0]:xr[1], yr[0]:yr[1]], genomeSet[x * k:(x + 1) * k, y * k:(y + 1) * k])

        idxs = np.indices(input.shape) # an array with one inner array per dimension descibing the respective index map
        idxFlat = zip(idxs[0].flat, idxs[1].flat) # flat version tupple if the array idexes per dimension
        return np.fromiter(map(cellIter, np.nditer(input), idxFlat),
                                  np.float64).reshape(input.shape)

    def convArea(self, iArea, gArea):
        """ Convolves an area of the input against an area of a genome """
        return np.sum(iArea*gArea)

    def haveSex(self, otherDude):
        baby = Dude(self.WH, self.kSize)

        """if random.randint(1,2)%2 == 0:
            mom = self
            dad = otherDude
        else:
            mom = otherDude
            dad = self

        sz = self.WH*self.kSize

        cutoff1 = random.randint(1,99)*kSize
        cutoff2 = random.randint(1, 99) * kSize

        baby.gen1[0:sz, 0:cutoff1] = mom.gen1[0:sz, 0:cutoff1]
        baby.gen1[0:sz, cutoff1:sz] = dad.gen1[0:sz, cutoff1:sz]
        baby.gen2[0:sz, 0:cutoff2] = mom.gen2[0:sz, 0:cutoff2]
        baby.gen2[0:sz, cutoff2:sz] = dad.gen2[0:sz, cutoff2:sz]

        return baby"""

        for y in range(0,self.WH):
            for x in range(0,self.WH):
                if (random.randint(1,2))%2 == 0:
                    sourceDude = self
                else:
                    sourceDude = otherDude

                k = self.kSize

                # Throw the dice and based on mutationRate apply mutation for an entire kernel area
                if (random.randint(1,1000)<=self.mutationRate):
                    baby.gen1[x*k:(x+1)*k,y*k:(y+1)*k] = self.normalizeSlice(np.random.random((self.kSize,self.kSize)))
                    baby.gen2[x*k:(x+1)*k,y*k:(y+1)*k] = self.normalizeSlice(np.random.random((self.kSize,self.kSize)))
                    baby.mutations = baby.mutations + 1
                else:
                    baby.gen1[x*k:(x+1)*k,y*k:(y+1)*k] = sourceDude.gen1[x*k:(x+1)*k,y*k:(y+1)*k]
                    baby.gen2[x*k:(x+1)*k,y*k:(y+1)*k] = sourceDude.gen2[x*k:(x+1)*k,y*k:(y+1)*k]

        return baby

##############################################################################################################3
