from random import randrange
from PIL.Image import open, fromarray
from numpy import asarray, uint8
import numpy as np
from heapq import heappop, heappush, heapify
from statistics import mean
import random
import math

def get_data(path):
    img = open(path)
    return asarray(img).tolist()


def split(data):
    left_data = []
    right_data = []
    for row in data:
        left_data.append(row[:round(len(row) / 2)])
        right_data.append(row[round(len(row) / 2):])
    return left_data, right_data


def combine(left_data, right_data):
    new_data = []
    for i in range(len(left_data)):
        new_data.append((np.concatenate((np.array(left_data[i]), np.array(right_data[i])))))
    return new_data


def representative_colors(data, num_colors):
    flattened = []
    for row in data:
        for pixel in row:
            flattened.append(pixel)
    centers = [(randrange(256), randrange(256), randrange(256)) for _ in range(num_colors)]
    prev_centers = []
    while centers != prev_centers:
        prev_centers = centers.copy()
        clusters = [[] for _ in range(num_colors)]
        for pixel in flattened:
            diffs = [abs(pixel[0] - center[0]) + abs(pixel[1] - center[1]) + abs(pixel[2] - center[2]) for center in
                     centers]
            clusters[diffs.index(min(diffs))].append(pixel)
        for i, cluster in enumerate(clusters):
            centers[i] = (round(mean(map(lambda rgb: rgb[0], cluster))), round(mean(map(lambda rgb: rgb[1], cluster))),
                          round(mean(map(lambda rgb: rgb[2], cluster))))
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


def simplify_colors(data, five_colors):
    recolored = []
    for row in data:
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


def to_image(data, savename):
    img = fromarray(asarray(data).astype(uint8))
    img.save(savename)
    img.show()


def similar_patches_locations(patch, gray_train_data):
    diffs_locations = []
    heapify(diffs_locations)
    for i, row in enumerate(gray_train_data):
        for j, pixel in enumerate(row):
            if not (i == 0 or i == len(gray_train_data) - 1 or j == 0 or j == len(row) - 1):
                a_patch = [gray_train_data[a][b] for a in [i - 1, i, i + 1] for b in [j - 1, j, j + 1]]
                diff = sum(list(map(lambda a: abs(a[0] - a[1]), zip(patch, a_patch))))
                if len(diffs_locations) < 6:
                    heappush(diffs_locations, (-diff, (i, j)))
                else:
                    # if a_patch is better than the worst patch in locations_diffs, sub it in
                    worst_diff_location = heappop(diffs_locations)
                    if diff < abs(worst_diff_location[0]):
                        heappush(diffs_locations, (-diff, (i, j)))
                    else:
                        heappush(diffs_locations, worst_diff_location)
    locations = []
    while len(diffs_locations) > 0:
        locations.append(heappop(diffs_locations)[1])
    return locations


def frequencies(colors):
    counts = {}
    color_tuples = [(color[0], color[1], color[2]) for color in colors]
    for color in color_tuples:
        if color not in counts.keys():
            counts[color] = 1
        else:
            counts[color] += 1
    return counts


def pick_color(counts, tiebreaker):
    max_count = max(counts.values())
    max_found = False
    best_color = None
    for color in counts:
        count = counts[color]
        if count == max_count:
            if max_found:
                return tiebreaker
            else:
                max_found = True
                best_color = color
    return list(best_color)


def basic_coloring(test_gray_data, train_gray_data, recolored_train_data):
    colored_test_data = []
    for i, row in enumerate(test_gray_data):
        new_row = []
        for j, pixel in enumerate(row):
            if i == 0 or i == len(test_gray_data) - 1 or j == 0 or j == len(row) - 1:
                new_row.append([0, 0, 0])
            else:
                patch = [test_gray_data[a][b] for a in [i - 1, i, i + 1] for b in [j - 1, j, j + 1]]
                locations = similar_patches_locations(patch, train_gray_data)
                colors = [recolored_train_data[i][j] for i, j in locations]
                counts = frequencies(colors)
                new_row.append(pick_color(counts, colors[-1]))  # location with min diff is at end of list
        colored_test_data.append(new_row)
        print(i / len(test_gray_data) * 100)
    return colored_test_data


def get_data_for_regression(gray_data, colored_data, rgb):
    X, y = [], []
    for i, row in enumerate(gray_data):
        if i != 0 and i != len(gray_data) - 1:
            for j, pixel in enumerate(row):
                if j != 0 and j != len(row) - 1:
                    it = zip([0, 0, 0, -1, 1], [0, -1, 1, 0, 0])
                    X_temp = []
                    X_temp.append(1)
                    for di, dj in it:
                        X_temp.append(gray_data[i + di][j + dj] / 255)
                        X_temp.append(math.pow(gray_data[i + di][j + dj] / 255, 2))
                    X.append(X_temp)
                    y.append(colored_data[i][j][rgb] / 255)
    return X, y


def f(data, weights):
    return np.dot(data, weights)


def loss(f, weights, data):
    a = [math.pow(f(x, weights) - y, 2) for x, y in data]
    return sum(a) / len(a)


def grad(f, weights, dp, res):
    return sum([(f(dp, weights) - res) * dp])


def find_weights(X, y):
    weights = np.zeros(11)
    X = np.array(X)
    alpha = 0.0005
    batch = 200
    for i in range(5000):
        idxs = [random.randrange(len(X)) for _ in range(batch)]
        gradient = weights[:]
        for dp, res in zip([X[idx] for idx in idxs], [y[idx] for idx in idxs]):
            gradient -= alpha * grad(f, weights, dp, res) * dp
        weights = gradient
    print("Loss: " + str(loss(f, weights, zip(X, y))))

    return weights


def improved_coloring(gray_data, red_model, green_model, blue_model):
    gray_data = np.array(gray_data)
    color = []
    for i, row in enumerate(gray_data):
        for j, pixel in enumerate(row):
            if i == 0 or i == len(gray_data) - 1 or j == 0 or j == len(row) - 1:
                color.extend([0, 0, 0])
            else:
                it = zip([0, 0, 0, -1, 1], [0, -1, 1, 0, 0])
                X_temp = []
                X_temp.append(1)
                for di, dj in it:
                    X_temp.append(gray_data[i + di][j + dj] / 255)
                    X_temp.append(math.pow(gray_data[i + di][j + dj] / 255, 3))

                color.append(min(max(f(X_temp, red_model) * 255, 0), 255))
                color.append(min(max(f(X_temp, green_model) * 255, 0), 255))
                color.append(min(max(f(X_temp, blue_model) * 255, 0), 255))

    color = np.array(color).reshape(len(gray_data), len(gray_data[0]), 3)
    return color
