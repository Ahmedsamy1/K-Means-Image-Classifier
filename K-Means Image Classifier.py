import numpy as np
import matplotlib.pyplot as plt
import random

def transformImageToBinary(img):
    binaryImage = np.array([])
    
    for i in range(784):
        if img[i] > 140:
            binaryImage = np.append(binaryImage, 1)
        else:
            binaryImage = np.append(binaryImage, 0)
    
    return binaryImage

def getDistBetImages(img1, img2):
    return np.linalg.norm(img1 - img2)

def getFurthestImage(src, img):
    furthest = src[0]
    dist1 = getDistBetImages(src[0], img)
    idx = 0
    
    for i in range(1, src.shape[0]):
        dist2 = getDistBetImages(src[i], img)
        if dist2 > dist1:
            furthest = src[i]
            dist1 = dist2
            idx = i
    
    return furthest, idx
 
def getBestCluster(allClusters, src):
    distances = []
    for i in range(30):
        distance = 0
        results = allClusters[i][0]
        means = allClusters[i][1]
        for j in range(2400):
            distance += getDistBetImages(src[j], means[int(results[j])])
        distances.append(distance)
    return distances.index(min(distances))

numberOfImages = 2400
numberOfClusters = 10

src = np.zeros(shape=(numberOfImages, 784))


for c in range(numberOfImages):
    img = plt.imread('Images/{}.jpg'.format(c+1))
    ser = np.array([])
    
    for i in range(28):
        for j in range(28):
            ser = np.append(ser, img[i][j])
    
    ser = transformImageToBinary(ser)
    src[c] = ser

allClusters = []
for iteration in range(30):
    print("Iteration", iteration + 1)
    means = np.zeros(shape=(numberOfClusters, 784))
    results = np.zeros(shape=(numberOfImages))
    copySrc = np.copy(src)
    
    randomNumber = random.randint(0, numberOfImages - 1)
    means[0] = src[randomNumber]
    copySrc = np.delete(copySrc, randomNumber, 0)
    means[1], idx = getFurthestImage(copySrc, means[0])
    copySrc = np.delete(copySrc, idx , 0)
    means[2], idx = getFurthestImage(copySrc, means[1])
    copySrc = np.delete(copySrc, idx , 0)
    means[3], idx = getFurthestImage(copySrc, means[2])
    copySrc = np.delete(copySrc, idx , 0)
    means[4], idx = getFurthestImage(copySrc, means[3])
    copySrc = np.delete(copySrc, idx , 0)
    means[5], idx = getFurthestImage(copySrc, means[4])
    copySrc = np.delete(copySrc, idx , 0)
    means[6], idx = getFurthestImage(copySrc, means[5])
    copySrc = np.delete(copySrc, idx , 0)
    means[7], idx = getFurthestImage(copySrc, means[6])
    copySrc = np.delete(copySrc, idx , 0)
    means[8], idx = getFurthestImage(copySrc, means[7])
    copySrc = np.delete(copySrc, idx , 0)
    means[9], _ = getFurthestImage(copySrc, means[8])
    
    while True:
        
        for i in range(numberOfImages):
            dist1 = getDistBetImages(src[i], means[0])
            results[i] = 0
            
            for j in range(1, numberOfClusters):
                dist2 = getDistBetImages(src[i], means[j])
                if dist2 < dist1:
                    dist1 = dist2
                    results[i] = j
        
        numOfImagesEachMean = np.zeros(shape=(numberOfClusters))
        newMeans = np.zeros(shape=(numberOfClusters, 784))
        
        for i in range(numberOfImages):
            numOfImagesEachMean[int(results[i])] += 1
            newMeans[int(results[i])] = np.add(newMeans[int(results[i])], src[i])
        
#        print(numOfImagesEachMean)
        
        for i in range(numberOfClusters):
            if numOfImagesEachMean[i] != 0:
                newMeans[i] = np.divide(newMeans[i], numOfImagesEachMean[i])
            else:
                newMeans[i] = means[i]
        
        if np.array_equal(newMeans, means):
            break
        
        means = newMeans
        
    tmpArray = []
    tmpArray.append(results)
    tmpArray.append(means)
    
    
    labels = np.loadtxt("Images/Training Labels.txt")
    
    numOfImagesPerCluster = np.zeros(shape=(numberOfClusters, numberOfClusters))
    classifiers = np.zeros(shape=(numberOfClusters))
    
    for i in range(numberOfImages):
        numOfImagesPerCluster[int(results[i])][int(labels[i])] += 1
                
    for i in range(numberOfClusters):
        classifiers[i] = np.where(numOfImagesPerCluster[i] == max(numOfImagesPerCluster[i]))[0][0]
    #    print("Cluster {} detects number {}".format(i, np.where(numOfImagesPerCluster[i] == max(numOfImagesPerCluster[i]))[0][0]))
    
    print(numOfImagesPerCluster)
    tmpArray.append(numOfImagesPerCluster)
    allClusters.append(tmpArray)
    
    
    plt.style.use('ggplot')
    
    x = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    for i in range(numberOfClusters):
        y[np.where(numOfImagesPerCluster[i] == max(numOfImagesPerCluster[i]))[0][0]] = max(numOfImagesPerCluster[i])
    
    x_pos = [i for i, _ in enumerate(x)]
    
    plt.bar(x_pos, y, color='green')
    plt.xlabel("Numbers")
    plt.ylabel("Count")
    plt.title("Number of digits in each cluster")
    
    plt.xticks(x_pos, x)
    
    plt.show()
    

bestClusterNumber = getBestCluster(allClusters, src)
print("Best iteration is ", bestClusterNumber + 1)

results = allClusters[bestClusterNumber][0]
means = allClusters[bestClusterNumber][1]
numOfImagesPerCluster = allClusters[bestClusterNumber][2]

plt.style.use('ggplot')

x = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(numberOfClusters):
    y[np.where(numOfImagesPerCluster[i] == max(numOfImagesPerCluster[i]))[0][0]] = max(numOfImagesPerCluster[i])

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, y, color='green')
plt.xlabel("Numbers")
plt.ylabel("Count")
plt.title("Number of digits in each cluster")

plt.xticks(x_pos, x)

plt.savefig('Counts.jpg', bbox_inches="tight")
plt.show()