import numpy as np
from scipy import misc
import random
import scipy.stats as st
from multiprocessing.dummy import Pool as ThreadPool

from neuretic import Dude


###############################################################################################################

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

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

pool = None

def lmap(func, *iterable):
    return list(pool.map(func, *iterable))


# *****************************************************************************************************
# *****************************************************************************************************
# *****************************************************************************************************

print("Starting ...")

noise = misc.imread('./res/NOISE.png')
test  = misc.imread('./res/TEST.png')
noise2D = rgb2linear(noise)
test2D = rgb2linear(test)

sampleSizeWidthAndHeight = len(noise[0])
alpha255 = np.array([255]*sampleSizeWidthAndHeight**2).reshape(sampleSizeWidthAndHeight, sampleSizeWidthAndHeight)
kSize = 11
totalPop = 21
totalPop = int(totalPop/2)*2

#plt.imshow(noise)
#noisr = linear2rgb(noise2D)
#plt.imshow(noisr)

probMatrix = [0]*511 + [1]*256 + [2]*128 + [3]*64 + [4]*32 + [5]*16 + [6]*8 + [7]*4 + [8]*2 + [9]*1
#probMatrix = [1]*10 + [2]*9 + [3]*8 + [4]*7 + [5]*6 + [6]*5 + [7]*4 + [8]*3 + [9]*2 + [10]*1

# Generate a population of dudes with random genomes
print("Generating startup population ...")
pop = []
for i in range(totalPop):
    pop.append(Dude(sampleSizeWidthAndHeight, kSize))

# Main loop
evolutionThreshold = 10
evolve = True
genNum = 1
while evolve:
    print("Running Generation "+str(genNum))
    # Apply the kernel to all Dudes in the population
    # This will also calculate the fit value for each Dude

    def popi(pi):
        print("Applying genome to Dude " + pi.name + ")...")
        pi.processAsync(noise2D, test2D)

    pool = ThreadPool(20)
    pool.map(popi, pop)
    pool.close()
    pool.join()

    '''for i in range(0,len(pop)):
        print("Applying genome to Dude "+ pop[i].name + "(" + str(i) +")...")
        pop[i].process1(noise2D, test2D)
        #print('Maps: ' + str(timeit.timeit('pop[i].process(noise2D, test2D)', "from __main__ import pop, i, noise2D, test2D", number=1)))
        #print('Olds: ' + str(timeit.timeit('pop[i].process1(noise2D, test2D)', "from __main__ import pop, i, noise2D, test2D", number=1)))
    '''

    # Sort by fitness
    pop.sort(key=lambda x: x.fitness)
    for i in range(0,len(pop)):
        print("Dude in pos " + str(i) + " ("+ pop[i].name +"): " + str(pop[i].fitness))

    matingList = []
    # Mate randonmly within the first half of individuals in the population
    for i in range(totalPop-1, totalPop//2-1, -1):
        pick1 = getRandomDude(totalPop, 0)
        pick2 = getRandomDude(totalPop, pick1)
        matingList.append((pick1,pick2))

    def mate(t):
        pick1 = t[0]
        pick2 = t[1]
        print("Mating " + pop[pick1].name + "(" + str(pick1) + ") and " + pop[pick2].name + "(" + str(pick2))
              #+ ") produced: " + pop[i].name + "(" + str(i) + ")" + (
              #    pop[i].mutations == 0 if "" else " MUTATED (" + str(pop[i].mutations) + ")"))
        return pop[pick1].haveSex(pop[pick2])


    pool = ThreadPool(10)
    babyList = list(pool.map(mate, matingList))
    pool.close()
    pool.join()

    pop[10:20] = babyList


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

