from PIL.Image import open, fromarray
from numpy import asarray, uint8
from heapq import heappop, heappush, heapify
from sklearn.cluster import KMeans


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
        new_data.append(left_data[i] + right_data[i])
    return new_data


def cluster(train_data):
    flattened = []
    for row in train_data:
        for pixel in row:
            flattened.append(pixel)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(asarray(flattened))
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


def show_image(data, savename):
    img = fromarray(asarray(data).astype(uint8))
    img.save(savename)
    img.show()


def similar_patches_locations(patch, train_gray_data):
    diffs_locations = []
    heapify(diffs_locations)
    for i, row in enumerate(train_gray_data):
        for j, pixel in enumerate(row):
            if not (i == 0 or i == len(train_gray_data) - 1 or j == 0 or j == len(row) - 1):
                a_patch = [train_gray_data[a][b] for a in [i - 1, i, i + 1] for b in [j - 1, j, j + 1]]
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


def color_in(test_gray_data, train_gray_data, recolored_train_data):
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


def get_X_y(gray_data, colored_data, rgb):
    X, y = [], []
    for i, row in enumerate(gray_data):
        if i != 0 and i != len(gray_data) - 1:
            for j, pixel in enumerate(row):
                if j != 0 and j != len(row) - 1:
                    X.append([round(gray_data[a][b] / 255, 2) for a in [i - 1, i, i + 1] for b in [j - 1, j, j + 1]])
                    y.append(round(colored_data[i][j][rgb] / 255, 2))
    return X, y


def model(X, y):
    return []  # TODO


def apply_models(gray_data, red_model, green_model, blue_model):
    return []  # TODO
