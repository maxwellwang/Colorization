from util import *

if __name__ == '__main__':
    # convert image into 2d array of pixels
    data = get_data('cherry.jpg')
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
    show_image(final_data)
