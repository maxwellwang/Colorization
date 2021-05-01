from PIL import Image as im
import numpy as np
from sklearn.cluster import KMeans


def get_data(path: str) -> list:
    img = im.open(path)
    return list(np.asarray(img))


def split(data):
    train_data = []
    test_data = []
    for row in data:
        train_data.append(row[:round(len(row) / 2)])
        test_data.append(row[round(len(row) / 2):])
    return train_data, test_data


def cluster(train_data):
    flattened = []
    for row in train_data:
        for pixel in row:
            flattened.append(pixel)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(np.asarray(flattened))
    centers = []
    for center in kmeans.cluster_centers_:
        centers.append([round(center[0]), round(center[1]), round(center[2])])
    return centers


def closest_color(pixel, five_colors):
    smallest_diff = 255 * 3
    closest = five_colors[0]
    for color in five_colors:
        diff = abs(pixel[0] - color[0]) + abs(pixel[1] - color[1]) + abs(pixel[2] - color[2])
        if diff < smallest_diff:
            closest = color
            smallest_diff = diff
    return closest


def recolor(train_data, five_colors):
    recolored = []
    for row in train_data:
        recolored.append([closest_color(pixel, five_colors) for pixel in row])
    return recolored


def get_gray_data(data):
    gray_data = []
    for row in data:
        gray_data_row = []
        for pixel in row:
            gray_data_row.append(round(0.21 * pixel[0] + 0.72 * pixel[1] + 0.07 * pixel[2]))
        gray_data.append(gray_data_row)
    return gray_data


def show_image(data):
    img = im.fromarray(np.asarray(data).astype(np.uint8))
    img.show()
