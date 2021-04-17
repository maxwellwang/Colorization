from PIL import Image
from numpy import asarray
from sklearn.cluster import KMeans


def get_data(path):
    img = Image.open(path)
    return asarray(img)


def split(data):
    train_data = []
    test_data = []
    for i in range(len(data)):
        train_row = []
        test_row = []
        for j in range(len(data[0])):
            if j <= len(data) / 2:
                train_row.append(data[i][j])
            else:
                test_row.append(data[i][j])
        train_data.append(train_row)
        test_data.append(test_row)
    return asarray(train_data), asarray(test_data)


def cluster(train_data):
    flattened = []
    for row in train_data:
        for pixel in row:
            flattened.append(pixel)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(asarray(flattened))
    centers = []
    for center in kmeans.cluster_centers_:
        centers.append([round(center[0]), round(center[1]), round(center[2])])
    return asarray(centers)


def closest_color(pixel, five_colors):
    smallest_diff = 255 * 3
    index = 0
    for i in range(len(five_colors)):
        color = five_colors[i]
        diff = abs(pixel[0] - color[0]) + abs(pixel[1] - color[1]) + abs(pixel[2] - color[2])
        if diff < smallest_diff:
            index = i
            smallest_diff = diff
    return five_colors[index]


def round_colors(train_data, five_colors):
    recolored = []
    for row in train_data:
        recolored_row = []
        for pixel in row:
            recolored_row.append(closest_color(pixel, five_colors))
        recolored.append(recolored_row)
    return asarray(recolored)


def get_gray_data(data):
    gray_data = []
    for row in data:
        gray_data_row = []
        for pixel in row:
            gray_data_row.append(round(0.21 * pixel[0] + 0.72 * pixel[1] + 0.07 * pixel[2]))
        gray_data.append(gray_data_row)
    return asarray(gray_data)
