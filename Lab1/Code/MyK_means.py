from collections import defaultdict
import random
from math import sqrt, exp
import csv
import numpy as np
from numpy import *
import time
import matplotlib.pyplot as plt
from Settings import *
import scipy

finalCenters = []
global_kernel_kind = 'linear_kernel'


def drawCategoryFigure(results, dataset, centers, dimension=2, df_data=[]):
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']
    drawX = dataset[:, 0]
    drawY = dataset[:, 1]
    centerX = [x[0] for x in centers]
    centerY = [x[1] for x in centers]
    if dimension == 2 or use_kernel_function_without_raising_dimension:
        fig = plt.figure()
        ax = fig.subplots()
        drawColors = []
        for index in range(len(results)):
            drawColors.append(colors[results[index]])
        ax.scatter(drawX, drawY, c=drawColors, alpha=0.5)
        if show_Kmeans_Center:
            ax.scatter(centerX, centerY, c='black', alpha=1)
        plt.title("cluster results:(centers are black)")
        plt.show()
    elif dimension == 3:
        ax2 = plt.axes(projection='3d')
        drawZ = dataset[:, 2]
        drawColors = []
        dataColors = []
        if show_Original_In_Higher_Dimension:
            dataColors = df_data['c']
        else:
            for index in range(len(results)):
                drawColors.append(colors[results[index]])
        for i in range(0, len(dataColors)):
            drawColors.append(colors[int(dataColors[i]) - 1])
        centerZ = [x[2] for x in centers]
        ax2.scatter3D(drawX, drawY, drawZ, c=drawColors, alpha=0.5)
        if not show_Original_In_Higher_Dimension:
            ax2.scatter3D(centerX, centerY, centerZ, c='black', alpha=1)
        plt.title("cluster results in higher dimension")
        plt.show()


def txt2csv(txtfilename, csvfilename):
    csvFile = open(csvfilename, mode='w', encoding='utf-8')
    writer = csv.writer(csvFile)
    with open(txtfilename, 'r') as f:
        for line in f:
            csvRow = line.split()
            writer.writerow(csvRow)
    csvFile.close()


def distance(a, b):
    dimensions = len(a)
    dimensionB = len(b)
    if dimensionB != dimensions:
        print("cannot calculate distance! two points have different dimensions!")
        return 0
    _sum = 0
    for dimension in range(dimensions):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return sqrt(_sum)


def Mahalanobis_Distance(x, x_i, x_j):
    xT = x.T
    D = np.cov(xT)
    invD = np.linalg.inv(D)  # The inverse of the covariance
    if use_python_library:
        return scipy.spatial.distance.mahalanobis(x_i, x_j, invD)
    else:
        tp = x_i - x_j
        return np.sqrt(np.dot(np.dot(tp.T, invD), tp))


# generate k cluster centers
def generate_k(dataset, k):
    '''
    generates random k points to be centers,return the centers list.
    '''
    centers = []
    length = len(dataset)
    dimension = len(dataset[0])
    for _k in range(k):
        center = []
        random_value = random.randint(0, length - 1)
        rand_point = dataset[random_value]
        for _d in range(0, dimension):
            center.append(rand_point[_d])
        centers.append(center)

    return centers


def generate_k_plus(dataset, k):
    '''
    1) generate a random point as the start point
    2) if a point has not be chosen, calculate the distance between it and the nearest chosen center
    3) chosen a new point as a new center, which the probability depends on the distance
    4) loop on 2) and 3) until k centers are chosen, then continue KMeans...
    '''
    centers = []
    length = len(dataset)
    dimension = len(dataset[0])
    for _k in range(k):
        center = []
        if _k == 0:
            random_value = random.randint(0, length - 1)
            random_point = dataset[random_value]
            for _d in range(0, dimension):
                center.append(random_point[_d])
        else:
            # tip: remember we need two values to accept the return value
            assignments, _IdontCare = assign_points(dataset, centers)
            distances = []
            probabilities = []
            sum_distance = 0
            # https://blog.csdn.net/yuzhihuan1224/article/details/100977580
            xMat = np.vstack((dataset, np.array(centers)))  # put 2 np.arrays together, for mahalanobis distance

            for assignment, point in zip(assignments, dataset):
                curDistance = distance(point, centers[assignment])
                if use_Mahalanobis_Distance:
                    curDistance = Mahalanobis_Distance(xMat, point, centers[assignment])
                if use_kernel_function_without_raising_dimension:
                    curDistance = gauss_kernel_distance(point, centers[assignment])
                distances.append(curDistance)
                sum_distance += curDistance
            for dis in distances:
                probabilities.append(dis / sum_distance)
            center = rate_random(dataset, probabilities)

        centers.append(center)
    return centers


def make_kernel2_3(x1, x2, method):
    if method == 'linear_kernel':
        a = x1 * x1
        b = x2 * x2
        c = sqrt(2) * x1 * x2
    elif method == 'gauss_kernel':
        sigmoid = gauss_kernel_sigmoid
        exp1 = exp(-sigmoid * x1 * x1)
        exp2 = exp(-sigmoid * x2 * x2)
        a = exp1 * exp2
        b = 2 * x1 * x2 * exp1 * exp2
        c = 2 * x1 * x1 * x2 * x2 * exp1 * exp2
    return a, b, c


def gauss_kernel_function(x1, x2):
    sigmoid = gauss_kernel_sigmoid
    dist = distance(x1, x2)
    res = exp(-dist*dist/(2*sigmoid*sigmoid))
    return res


def gauss_kernel_distance(x1, x2):
    sum1 = gauss_kernel_function(x1, x1)
    sum2 = gauss_kernel_function(x1, x2)
    sum3 = gauss_kernel_function(x2, x2)
    return sqrt(sum1-2*sum2+sum3)


def kernel_process(datas_list, method):
    dimension = len(datas_list[0])
    newdatas_list = np.zeros((datas_list.shape[0], dimension + 1))
    length = len(datas_list)
    if dimension > 2:
        return
    # https://www.jianshu.com/p/c9825e9be248
    for index in range(0, length):
        tmpdata = datas_list[index]
        a, b, c = make_kernel2_3(tmpdata[0], tmpdata[1], method)
        newdatas_list[index] = [a, b, c]
    return newdatas_list


def rate_random(datas_list, rates_list):
    start = 0
    random_num = random.random()
    dimension = len(datas_list[0])
    result = []
    for idx, score in enumerate(rates_list):
        start += score
        if random_num <= start:
            break

    random_point = datas_list[idx]
    for i in range(0, dimension):
        result.append(random_point[i])
    return result


def assign_points(data_points, centers):
    assignments = []
    sumDistance = 0
    # https://blog.csdn.net/yuzhihuan1224/article/details/100977580
    xMat = np.vstack((data_points, np.array(centers)))  # put 2 np.arrays together, for mahalanobis distance
    for point in data_points:
        shortest = float('inf')  # positive infinity
        shortest_index = 0
        # calculate the distance between cur point and each center,
        # find the shortest result and mark cur point with the center's index
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if use_Mahalanobis_Distance:
                val = Mahalanobis_Distance(xMat, point, centers[i])
            if use_kernel_function_without_raising_dimension:
                val = gauss_kernel_distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)

    # calculate the sumDistance
    tempZip = zip(assignments, data_points)
    for assignment, data_point in tempZip:
        if use_Mahalanobis_Distance:
            sumDistance += Mahalanobis_Distance(xMat, data_point, centers[assignment])
        elif use_kernel_function_without_raising_dimension:
            sumDistance += gauss_kernel_distance(data_point, centers[assignment])
        else:
            sumDistance += distance(data_point, centers[assignment])

    return assignments, sumDistance


def update_centers(dataset, assignments):
    global finalCenters
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, dataset):
        new_means[assignment].append(point)

    for points in new_means.values():  # calculate mean values
        centers.append(point_avg(points))
    finalCenters = centers
    return centers


def point_avg(points):
    '''
    return a new point which shows the center of the input points
    '''
    dimensions = len(points[0])

    new_center = []

    for dimension in range(dimensions):
        dim_sum = 0  # dimension sum
        for p in points:
            dim_sum += p[dimension]

        # average of each dimension
        new_center.append(dim_sum / float(len(points)))

    return new_center


class MyKMeans(object):
    def __init__(self, n_clusters, max_iter, n_init, mode):
        self.inertia_ = 0
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.mode = mode

    def kMeans(self, dataset):
        initnum = 0
        iternum = 0
        minDis = float('inf')
        assignments = []
        old_assignments = []

        if self.mode == "kmeans_kernel":
            if not use_kernel_function_without_raising_dimension:
                tmpdataset = kernel_process(dataset, global_kernel_kind)
                dataset = tmpdataset

        while initnum < self.n_init:
            if self.mode == "kmeans":
                k_points = generate_k(dataset, self.n_clusters)
            elif self.mode == "kmeans++":
                k_points = generate_k_plus(dataset, self.n_clusters)
            elif self.mode == "kmeans_kernel":
                k_points = generate_k(dataset, self.n_clusters)
            calAssignments, sumDistance = assign_points(dataset, k_points)
            if sumDistance < minDis:
                minDis = sumDistance
                assignments = calAssignments
            initnum += 1

        start = time.time()
        self.inertia_ += minDis

        while iternum < self.max_iter and assignments != old_assignments:
            new_centers = update_centers(dataset, assignments)
            old_assignments = assignments
            assignments, sumDistance = assign_points(dataset, new_centers)
            iternum += 1
        # https://www.runoob.com/python/python-func-zip.html
        if show_Iter_Num:
            print("the number of iternum is: ", iternum)

        self.inertia_ = sumDistance  # fix bug #20220918_1345
        if Only_Calculate_The_first_Centers_Choose:
            self.inertia_ = minDis
        end = time.time()
        if show_Time_Use:
            print("total category time is: ", end - start, "(s)")
        if show_Kmeans_Center:
            returnCenters = finalCenters
        else:
            returnCenters = []

        return dataset, assignments, returnCenters
