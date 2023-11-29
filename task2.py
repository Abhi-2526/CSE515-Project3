from math import ceil
import numpy as np
from torchvision.models.resnet import *
import torchvision
import torch
import matplotlib
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from scipy.spatial.distance import directed_hausdorff
from sklearn.manifold import MDS
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# load caltech dataset
dataset = torchvision.datasets.Caltech101("../CSE515-Project3/",
                                          download=True)  # set download to true if it is not already in the folder
labels_caltech101 = np.array([dataset[i][1] for i in range(len(dataset))])
downdata_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)
label_name_to_idx = {name: idx for idx, name in enumerate(dataset.categories)}

# load necessary data
FCEvenData = torch.load("./FCEvenData.pt")  # even ResNet FC data
FCOddData = torch.load("./FCOddData.pt")  # odd ResNet FC data
evenImageLabelList = []
for i in range(int(ceil(len(labels_caltech101) / 2))):
    evenImageLabelList.append(labels_caltech101[2 * i])
# FCdissimilarityMatrix = torch.load("./FCdissimilarityMatrix.pt")
fcMDS = torch.load("./fcMDS.pt")
relaxed_fcCluster_calculated = torch.load("./fcClusters_Calculated.pt")
corePointsFC = torch.load("./T2corePoints.pt")


def compare_features(query_features, database_features):
    dot_product = np.dot(query_features, database_features)
    norm_a = np.linalg.norm(query_features)
    norm_b = np.linalg.norm(database_features)
    similarity = dot_product / (norm_a * norm_b)
    return similarity


def get_label_key_for_id(id):
    return list(label_name_to_idx.keys())[list(label_name_to_idx.values()).index(id)]


# compute dissimilarity matrix for MDS
def computeDissimilarityMatrix(data):
    matrix = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            matrix[i][j] = directed_hausdorff([data[i]], [data[j]])[0]
            # matrix[i][j] = compare_features(data[i], data[j])
    return matrix


# calculate MDS 2D coordinates
def calculatePlotMDS(dm):
    embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=0, normalized_stress=False)
    # embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
    X_transformed = embedding.fit_transform(dm)

    return X_transformed


# calculate core points to query on (already saved in T2corePoints.pt)
def calculateCorePoints(D, labels):
    i = 0
    corePoints = []
    for label in range(101):
        sum = np.zeros((1000))
        count = 0
        while i < len(labels) and labels[i] == label:
            sum += D[i]
            i += 1
            count += 1
        corePoints.append(sum / count)
    return corePoints


# corePointsFC = calculateCorePoints(FCEvenData, evenImageLabelList)
# torch.save(corePointsFC, "T2corePoints.pt")


'''
DBScan algorithm code
'''


def dbscanForLabel(D, eps, MinPts, corePoints, labelList):
    clusters = []
    C = 0
    # start at corePoint
    for corepoint in corePoints:
        clusters.append([])
        NeighborPts = region_query_for_corePoint(D, corepoint, eps)
        grow_cluster_for_corepoints(D, clusters, corepoint, NeighborPts, C, eps, MinPts, labelList)
        C += 1

    return clusters


def grow_cluster_for_corepoints(D, clusters, P, NeighborPts, C, eps, MinPts, labelList):
    # Assign the cluster label to the seed point.
    i = 0
    while i < len(NeighborPts):

        # Get the next point from the queue.
        Pn = NeighborPts[i]
        clusters[C].append(NeighborPts[i])

        PnNeighborPts = region_query_for_corePoint(D, Pn, eps)

        if len(PnNeighborPts) >= MinPts:
            # NeighborPts = NeighborPts + PnNeighborPts
            NeighborPts = NeighborPts + list(set(PnNeighborPts) - set(NeighborPts))
        else:
            NeighborPts = NeighborPts + list(set(relax_dbscan(C, labelList)) - set(NeighborPts))
            # Advance to the next point in the FIFO queue.
        i += 1


def relax_dbscan(label, labelList):
    # relax dbscan add more neighbor points to query by
    points = []
    for i in range(len(labelList)):
        if labelList[i] == label and len(points) < 10:
            points.append(i)
    return points


def region_query_for_corePoint(D, corePoint, eps):
    # Find all points in dataset `D` within distance `eps` of point `P`.
    neighbors = []

    for Pn in range(0, len(D)):
        similarity = compare_features(corePoint, D[Pn])

        # since using cosine similarity, similary > eps
        if similarity > eps:
            neighbors.append(Pn)

    return neighbors


# relaxed_fcCluster_calculated = dbscanForLabel(FCEvenData, 0.8, 10, corePointsFC, evenImageLabelList)
# torch.save(relaxed_fcCluster_calculated, "fcClusters_Calculated.pt")


'''
Plot image and 2D graph code
'''


# plot image thumbnails
def plotImageThumbnails(imageIDList):
    fig = plt.figure(figsize=(25, 10 * ceil(len(imageIDList)) / 15))
    plt.axis('off')
    for i in range(len(imageIDList)):
        fig.add_subplot(ceil(len(imageIDList) / 15), 15, i + 1)
        im = np.asarray(dataset[2 * imageIDList[i]][0])
        plt.axis('off')
        plt.imshow(im)
        plt.title("id = " + str(2 * imageIDList[i]), fontsize=10)
    plt.show()


# plot 2D MDS graph
def plotClusters(points, labels, title):
    fig = plt.figure(2, (15, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax = sns.scatterplot(x=points[:, 0], y=points[:, 1],
                         hue=labels, palette=sns.color_palette("tab10"))
    plt.title(title)
    plt.show()


# helper functions
def getCoordinateList(indexList, coordinateList):
    pointList = coordinateList[indexList[0]]
    for i in range(1, len(indexList)):
        pointList = np.vstack((pointList, coordinateList[indexList[i]]))
    return pointList


def getPlotLabelList(clusterList):
    labelList = []
    j = 0
    for i in clusterList:
        labelList = labelList + [j] * len(i)
        j += 1
    return labelList


def getClusterCoordinateList(clusterList, coordinateList):
    result = getCoordinateList(clusterList[0], coordinateList)
    for i in range(1, len(clusterList)):
        result = np.vstack((result, getCoordinateList(clusterList[i], coordinateList)))
    return result


def getLabelList(labelIndexList, label):
    result = []
    for i in range(len(labelIndexList)):
        if labelIndexList[i] == label:
            result.append(i)
    return result


# calculate accuracy scores
def calculateLabelPrecision(label, cluster, labelIndexList):
    # true postives/(true postitives + false positives)
    labelList = getLabelList(labelIndexList, label)
    truePositives = list(set(labelList) & set(cluster))
    precision = len(truePositives) / len(cluster)
    return precision


def calculateLabelRecall(label, cluster, labelIndexList):
    # true postives/(true postitives + false positives)
    labelList = getLabelList(labelIndexList, label)
    truePositives = list(set(labelList) & set(cluster))
    falseNegatives = list(set(labelList) - set(cluster))
    recall = len(truePositives) / (len(truePositives) + len(falseNegatives))
    return recall


def calculateF1Score(label, cluster, labelIndexList):
    prec = calculateLabelPrecision(label, cluster, labelIndexList)
    recall = calculateLabelRecall(label, cluster, labelIndexList)
    if (prec + recall) != 0:
        f1Score = 2 * (prec * recall) / (prec + recall)
    else:
        f1Score = 0
    return f1Score


def calcuateLabelAccuracy(label, cluster, labelIndexList):
    # number of correct predictions / size of dataset
    labelList = getLabelList(labelIndexList, label)
    truePositives = list(set(labelList) & set(cluster))
    falseNegatives = list(set(labelList) - set(cluster))
    falsePositives = list(set(cluster) - set(labelList))
    numTrueNegatives = len(labelIndexList) - len(truePositives) - len(falseNegatives) - len(falsePositives)
    accuracy = (len(truePositives) + numTrueNegatives) / len(labelIndexList)
    return accuracy


def computeRelevantClusters(c, label):
    data = torch.load("./FCEvenData.pt")
    # Even ID label
    evenImageLabelList = []
    for i in range(int(ceil(len(labels_caltech101) / 2))):
        evenImageLabelList.append(labels_caltech101[2 * i])
    # print(len(evenImageLabelList))

    clusters = torch.load("./fcClusters_Calculated.pt")
    corePoints = torch.load("./T2corePoints.pt")

    dist = {}
    # initial core point for that label
    for i in range(len(corePoints)):
        dist.update({i: compare_features(corePoints[i], corePoints[label])})

    sortedDict = dict(sorted(dist.items(), key=lambda item: item[1]))

    queryPts = []
    idx = 0
    while len(queryPts) < c:
        queryPts.append(list(sortedDict)[len(corePoints) - 1 - idx])
        idx += 1

    c_clusters = []
    for i in range(0, c):
        c_clusters.append(clusters[queryPts[i]])

    return c_clusters


def getClusterLabel(cluster):
    clusters = torch.load("./fcClusters_Calculated.pt")
    i = 0
    for clust in clusters:
        if cluster == clust:
            return i
        i += 1


def task_2a(c, l):
    print("\nLabel ID= " + str(l) + " (" + list(label_name_to_idx.keys())[
        list(label_name_to_idx.values()).index(l)] + ")")
    evenImageLabelList = []
    for i in range(int(ceil(len(labels_caltech101) / 2))):
        evenImageLabelList.append(labels_caltech101[2 * i])

    clusters = computeRelevantClusters(c, l)
    print(str(c) + " most relevant clusters:")
    i = 0

    # doesn't seem like we need to compute accuracy for the predictions
    # for clust in clusters:
    # i += 1
    # print("\nCluster: " + str(i))
    # print([x * 2 for x in clust])
    # print("Accuracy: " + str(calcuateLabelAccuracy(l, clust, evenImageLabelList)))
    for clust in clusters:
        plotImageThumbnails(clust)

        # fcMDS = torch.load("./fcMDS.pt")
        # plotClusters(getClusterCoordinateList(clusters, fcMDS), getPlotLabelList(clusters), "Clusters for label " + str(l))

    d = FCEvenData[clusters[0][0]]
    for i in range(1, len(clusters[0])):
        d = np.vstack((d, FCEvenData[clusters[0][i]]))

    for clust in range(1, len(clusters)):
        for i in range(len(clusters[clust])):
            d = np.vstack((d, FCEvenData[clusters[clust][i]]))

    # print(d.shape)
    dm = computeDissimilarityMatrix(d)
    mds = calculatePlotMDS(dm)
    lab = []
    for clust in range(len(clusters)):
        for i in range(len(clusters[clust])):
            lab.append(clust)
    print("number of points: " + str(len(lab)))
    title = "\nLabel ID= " + str(l) + " (" + list(label_name_to_idx.keys())[
        list(label_name_to_idx.values()).index(l)] + ")"
    plotClusters(mds, lab, title)


def task_2b(id, c):
    if id % 2 == 0:
        print("Please enter an even ID")
    else:
        clusters = computeRelevantClusters(c, labels_caltech101[id])
        i = 0
        print(str(c) + " most relevant labels for image id = " + str(id))
        for clust in clusters:
            i += 1
            label = getClusterLabel(clust)
            print(str(i) + ". " + list(label_name_to_idx.keys())[list(label_name_to_idx.values()).index(label)])


def task_2c():
    total = 0
    for i in range(len(relaxed_fcCluster_calculated)):
        print("\nLabel " + str(i) + " (" + list(label_name_to_idx.keys())[
            list(label_name_to_idx.values()).index(i)] + ")")
        total += calcuateLabelAccuracy(i, relaxed_fcCluster_calculated[i], evenImageLabelList)
        print("Precision: " + str(calculateLabelPrecision(i, relaxed_fcCluster_calculated[i], evenImageLabelList)))
        print("Recall: " + str(calculateLabelRecall(i, relaxed_fcCluster_calculated[i], evenImageLabelList)))
        print("F1 Score: " + str(calculateF1Score(i, relaxed_fcCluster_calculated[i], evenImageLabelList)))

    print("\nOverall Accuracy: " + str(total / len(relaxed_fcCluster_calculated)))


def run_task2():
    while True:
        choice = str(input("Please enter the task you want to execute (2a/2b/2c): "))
        if choice == "2a":
            c = int(input("Please enter number of relevant clusters you want: "))
            l = int(input("Please enter the label ID you would like to create clusters for: "))
            task_2a(c, l)
            break
        elif choice == "2b":
            id = int(input("Please enter the odd image ID you would like to visualize clusters for: "))
            while id % 2 != 1:
                id = int(input("Please enter an ODD image ID: "))
            c = int(input("Please enter the number of relevant lables you want: "))
            task_2b(id, c)
            break
        elif choice == "2c":
            task_2c()
            break
        else:
            print("Invalid input. Please choose 2a/2b/2c")