import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import misc
from scipy import signal as sg
import random
import array
import scipy.stats as st




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
        level2 = self.processGenomeSet(self.gen2, level1)
        self.fitness = self.getFitness(level2, test)

        self.output = level2

        return level2

    def getFitness(self, output, test):
        return abs(output-test).sum()/((self.WH*self.kSize)**2)

    def getBounds(self, v, k, s):
        if 0 > v - k//2:
            return [0, k]
        if s <= v + k//2:
            return [s-k,s]
        return [v-k//2, v + k//2 + 1]

    def processGenomeSet(self, genomeSet, input):
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

                # Throw the dice and based on mutationRate apply mutation for an entire kernel area
                if (random.randint(1,1000)<=self.mutationRate):
                    baby.gen1[x*kSize:(x+1)*kSize,y*kSize:(y+1)*kSize] = self.normalizeSlice(np.random.random((self.kSize,self.kSize)))
                    baby.gen2[x*kSize:(x+1)*kSize,y*kSize:(y+1)*kSize] = self.normalizeSlice(np.random.random((self.kSize,self.kSize)))
                    baby.mutations = baby.mutations + 1
                else:
                    baby.gen1[x*kSize:(x+1)*kSize,y*kSize:(y+1)*kSize] = sourceDude.gen1[x*kSize:(x+1)*kSize,y*kSize:(y+1)*kSize]
                    baby.gen2[x*kSize:(x+1)*kSize,y*kSize:(y+1)*kSize] = sourceDude.gen2[x*kSize:(x+1)*kSize,y*kSize:(y+1)*kSize]

        return baby

##############################################################################################################3



def rgb2linear(img):
    """ Coverts an rgb matrix into a 2D matrix where each cell represents the combined 24bit rgb color in one number """
    sampleSizeW_H = len(img[0])
    img2D = img[:, :, 0] * 2 ** 16 + img[:, :, 1] * 2 ** 8 + img[:, :, 2]
    return img2D.reshape(1, sampleSizeW_H, sampleSizeW_H).reshape(sampleSizeW_H, sampleSizeW_H)

def linear2rgb(img):
    """ Coverts an rgb matrix into a 2D matrix where each cell represents the combined 24bit rgb color in one number """
    sampleSizeW_H = len(img[0])

    c1 = img//2**16
    c2 = (img-c1*2**16)//2**8
    c3 = img-(c1*2**16 + c2*2**8)

    r = np.empty((sampleSizeW_H, sampleSizeW_H, 4))
    r[:, :, 0] = c1
    r[:, :, 1] = c2
    r[:, :, 2] = c3
    r[:, :, 3] = alpha255

    r = np.array(r//1).astype('uint8')

    return r

def getRandomDude(totalPopNum, excludeNum):
    notPicked = True
    while notPicked:
        randIndex = random.randint(0,len(probMatrix)-1)
        pick = probMatrix[randIndex]
        if pick != excludeNum:
            notPicked = False
    return pick


# *****************************************************************************************************
# *****************************************************************************************************
# *****************************************************************************************************

print("Starting ...")

noise = misc.imread('/home/me/Documents/Jupyter/NOISE.png')
test = misc.imread('/home/me/Documents/Jupyter/TEST.png')
noise2D = rgb2linear(noise)
test2D = rgb2linear(test)

sampleSizeWidthAndHeight = len(noise[0])
alpha255 = np.array([255]*sampleSizeWidthAndHeight**2).reshape(sampleSizeWidthAndHeight, sampleSizeWidthAndHeight)
kSize = 11
totalPop = 21
totalPop = int(totalPop/2)*2

plt.imshow(noise)
noisr = linear2rgb(noise2D)
plt.imshow(noisr)

probMatrix = [0]*511 + [1]*256 + [2]*128 + [3]*64 + [4]*32 + [5]*16 + [6]*8 + [7]*4 + [8]*2 + [9]*1
#probMatrix = [1]*10 + [2]*9 + [3]*8 + [4]*7 + [5]*6 + [6]*5 + [7]*4 + [8]*3 + [9]*2 + [10]*1

# Generate a population of dudes with random genomes
print("Generating startup population ...")
pop = []
for i in range(totalPop):
    pop.append(Dude(sampleSizeWidthAndHeight, kSize))

# Generate a Gaussian Kernel to apply to the 2D Genomes
#gKern = gkern(21)

evolutionThreshold = 10
evolve = True
genNum = 1
while evolve:
    print("Running Generation "+str(genNum))
    # Apply the kernel to all Dudes in the population
    # This will also calculate the fit value for each Dude
    for i in range(0,len(pop)):
        print("Applying genome to Dude "+ pop[i].name + "(" + str(i) +")...")
        pop[i].process(noise2D, test2D)

    # Sort by fitness
    pop.sort(key=lambda x: x.fitness)
    for i in range(0,len(pop)):
        print("Dude in pos " + str(i) + " ("+ pop[i].name +"): " + str(pop[i].fitness))

    # Mate randonmly within the first half of individuals in the population
    for i in range(totalPop-1, totalPop//2-1, -1):
        pick1 = getRandomDude(totalPop, 0)
        pick2 = getRandomDude(totalPop, pick1)
        pop[i] = pop[pick1].haveSex(pop[pick2])
        print("Mating " + pop[pick1].name + "(" + str(pick1) + ") and " + pop[pick2].name + "(" + str(pick2) + ") produced: " + pop[i].name + "(" + str(i) + ")" + (pop[i].mutations == 0 if "" else " MUTATED ("+ str(pop[i].mutations) +")"))

    bestFitnessSoFar = pop[0].fitness
    print("\nBest fitnes so far: " + pop[0].name + ": " + str(bestFitnessSoFar) + "\n")

    output = linear2rgb(pop[0].output)
    misc.imsave("/home/me/Documents/Jupyter/output-gen"+str(genNum)+".png", output)

    if bestFitnessSoFar < evolutionThreshold:
        evolve = False
    genNum = genNum + 1

#plt.imshow(gKern, interpolation='none')

stop = 1
#plt.imshow(noise)