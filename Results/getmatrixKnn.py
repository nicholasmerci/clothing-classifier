import numpy as np
from matplotlib import pyplot as plt

def getMatrix(file):
    contents = open(file).read()
    contents = contents.replace('[','')
    lines = contents.split(']')

    # Rimuovo liste vuote
    lines = [x for x in lines if len(x)!=0]

    # Casto
    lines = [i.split() for i in lines]
    
    '''for line in lines:
        print(line)'''

    x = np.array(lines)
    x = x.astype(float)

    return x

def saveConfusionMatrix(matrix, photoName):

    plt.clf()
    plt.imshow(matrix)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12],["0","1","2","3","4","5","6","7","8","9","10","11","12"])
    plt.yticks([0,1,2,3,4,5,6,7,8,9,10,11,12], ["0","1","2","3","4","5","6","7","8","9","10","11","12"])
    plt.ylabel("Real")
    #plt.show()

    plt.savefig(photoName +'.jpg')

#saveConfusionMatrix(getMatrix('./cosine1.txt'), 'cosine1')

def readEntireFile(file, prefix):
    contents = open(file).read()
    elems = contents.split("K=")

    # Pulisco
    elems = [x for x in elems if len(x)!=0]

    #print(len(elems))
    for elem in elems:
        elem = elem.replace('[','').replace(']','')
        righe = elem.split('\n')
        k = righe[0]

        print("k = " + k)
        matrice = righe[6:19]
        print(matrice)
        #lines = contents.split(']')

        lines = [i.split() for i in matrice]
    
        '''for line in lines:
            print(line)'''

        x = np.array(lines)
        x = x.astype(float)
        print("size: " + str(x.shape))
        #print(x)
        saveConfusionMatrix(x, prefix + "_" + k)


readEntireFile("svm/linear.txt", "linear")    