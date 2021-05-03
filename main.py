from util import *


def basic_agent(filename, savename):
    # convert image into 2d array of pixels
    data = get_data(filename)
    # split left side into train and right side into test
    train_data, test_data = split(data)
    # get the 5 central colors on train side
    five_colors = cluster(train_data)
    # recolor the train side using the 5 central colors
    recolored_train_data = recolor(train_data, five_colors)
    # get gray version of data
    gray_data = get_gray_data(data)
    # split left side into train and right side into test
    train_gray_data, test_gray_data = split(gray_data)
    # color in test gray data using recolored train data and train gray data
    colored_test_data = color_in(test_gray_data, train_gray_data, recolored_train_data)
    # combine recolored train data with colored test data
    final_data = combine(recolored_train_data, colored_test_data)
    show_image(final_data, savename)


def improved_agent(filename, savename):
    # convert image into 2d array of pixels
    data = get_data(filename)
    # split left side into train and right side into test
    train_data, test_data = split(data)
    # get gray version of data
    gray_data = get_gray_data(data)
    # split left side into train and right side into test
    train_gray_data, test_gray_data = split(gray_data)
    # get the input and output data that we will use to train our model
    red_X, red_y = get_X_y(train_gray_data, train_data, 0)
    green_X, green_y = get_X_y(train_gray_data, train_data, 1)
    blue_X, blue_y = get_X_y(train_gray_data, train_data, 2)
    # train the models
    red_model = model(red_X, red_y)
    green_model = model(green_X, green_y)
    blue_model = model(blue_X, blue_y)
    # apply the models to the data
    final_data = apply_models(gray_data, red_model, green_model, blue_model)
    show_image(final_data, savename)


if __name__ == '__main__':
    # basic_agent('cherry.jpg', 'basic_agent_result.jpg')
    improved_agent('cherry.jpg', 'improved_agent_result.jpg')
