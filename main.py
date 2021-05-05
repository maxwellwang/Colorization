from util import *


def basic_agent(filename, savename):
    # convert image into 2d list of pixels
    data = get_data(filename)
    # split data into left train and right test sides
    train_data, test_data = split(data)
    # get the 5 most representative colors on train side
    five_colors = representative_colors(train_data, 5)
    # recolor the train side using the 5 most representative colors
    recolored_train_data = simplify_colors(train_data, five_colors)
    # transform the data into grayscale
    gray_data = get_gray_data(data)
    # split data into left train and right test sides
    gray_train_data, gray_test_data = split(gray_data)
    # color in gray test data by finding similar patches in gray train data and using the corresponding colors from
    # recolored train data
    colored_test_data = basic_coloring(gray_test_data, gray_train_data, recolored_train_data)
    # combine recolored train data with newly colored in test data
    final_data = combine(train_data, colored_test_data)
    to_image(final_data, savename)


def improved_agent(filename, savename):
    # convert image into 2d list of pixels
    data = get_data(filename)
    # split data into left train and right test sides
    train_data, test_data = split(data)
    # transform the data into grayscale
    gray_data = get_gray_data(data)
    # split data into left train and right test sides
    gray_train_data, gray_test_data = split(gray_data)
    # get the input and output data that we will use to train our model
    red_X, red_y = get_data_for_regression(gray_train_data, train_data, 0)
    green_X, green_y = get_data_for_regression(gray_train_data, train_data, 1)
    blue_X, blue_y = get_data_for_regression(gray_train_data, train_data, 2)
    # train the models
    red_model = find_weights(red_X, red_y)
    green_model = find_weights(green_X, green_y)
    blue_model = find_weights(blue_X, blue_y)
    # apply the models to the data
    final_data = improved_coloring(gray_test_data, red_model, green_model, blue_model)
    to_image(combine(train_data, final_data), savename)
    # print(red_model)
    # print(green_model)
    # print(blue_model)


if __name__ == '__main__':
    # basic_agent('cherry.jpg', 'basic_cherry.jpg')
    improved_agent('mountain.png', 'improved_mountain_f.png')
