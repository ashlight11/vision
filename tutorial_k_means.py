from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import numpy as np

#Ensuite charger une image et la convertir de BGR à RGB si nécessaire et l’afficher :
image = cv2.imread('nuka_cola.jpg')
image = cv2.resize(image, (400, 800))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
fig, axs = plt.subplots(2)
axs[0].axis("off")
axs[0].imshow(image)


n_clusters=5
image = image.reshape((image.shape[0] * image.shape[1], 3))
clt = KMeans(n_clusters = n_clusters )
clt.fit(image)

def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    return bar

hist = centroid_histogram(clt)
bar = plot_colors(hist, clt.cluster_centers_)
axs[1].axis("off")
axs[1].imshow(bar)
plt.show()
