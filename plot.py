'''
    plot the pressure point on screen
'''

import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import time
from matsense.tools import load_config, make_action
from matsense.uclient import Uclient
from matsense.process import Processor

BASE_DATA_DIR = "./data/"
LETTER_IN_KEYBOARD = [['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', ''],
                      ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ''],
                      ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '']]


def mock_generator():
    while True:
        yield 0.5, 0.5, 1  # mock the data generated from processor


def file_generator(filename):
    with open(filename, "r") as f:
        data = f.read().split("\n")
    for line in data:
        line = json.loads(line)
        yield line[0], line[1], line[2]


def ave_cali_generator(generator):
    while True:
        data = next(generator)
        for i, row in enumerate(data):
            for j, line in enumerate(row):
                if (cali_data[i][j] != 0):
                    data[i][j] = line / cali_data[i][j]
        yield data


def cali_each_point():
    print("begin calibrate, please move your tongue")
    cali_time = 20
    begin_time = time.time()
    max_pressure = []
    for i in range(13):
        max_pressure.append([])
    for i in range(13):
        for j in range(14):
            max_pressure[i].append(0)
    while (time.time() - begin_time < cali_time):
        raw_data = next(my_processor.gen_wrapper(tongue_client.gen()))
        for i, row in enumerate(max_pressure):
            for j, line in enumerate(row):
                if line < raw_data[i][j]:
                    max_pressure[i][j] = raw_data[i][j]
        time.sleep(0.02)
    print(max_pressure)
    global cali_data
    cali_data = max_pressure
    return max_pressure


def draw_border():
    # all organized as point
    row = np.array([[[0.1, 1], [0.9, 1]], [[0.1, 0.7], [0.9, 0.7]], [
                   [0.2, 0.3], [0.8, 0.3]], [[0.35, 0], [0.65, 0]]])
    col_1 = np.array([[[i * (0.9 - 0.1) / 10 + 0.1, 0.7],
                     [i * (0.9 - 0.1) / 10 + 0.1, 1]] for i in range(11)])
    col_2 = np.array([[[i * (0.8 - 0.2) / 9 + 0.2, 0.3],
                     [i * (0.8 - 0.2) / 9 + 0.2, 0.7]] for i in range(10)])
    col_3 = np.array([[[i * (0.65 - 0.35) / 7 + 0.35, 0],
                     [i * (0.65 - 0.35) / 7 + 0.35, 0.3]] for i in range(8)])
    col = [col_1, col_2, col_3]
    for line in row:
        plt.plot(line[:, 0], line[:, 1], color='#000000')
    gap_lengths = [(0.9 - 0.1) / 10, (0.8 - 0.2) / 9, (0.65 - 0.35) / 7]
    for i, c in enumerate(col):
        gap_length = gap_lengths[i]
        for j, line in enumerate(c):
            plt.plot(line[:, 0], line[:, 1], color='#000000')
            # 0.01 for half size of the letter
            plt.text(line[0][0] + gap_length / 2 - 0.01,
                     np.average([line[0][1], line[1][1]]), LETTER_IN_KEYBOARD[i][j])
    plt.show()


def scatter_plot():

    plt.ion()

    for _ in my_generator:
        # clean previous picture
        plt.cla()
        draw_border()

        row, col, val = next(my_generator)

        # record all data
        if (output_filename != None):
            with open(BASE_DATA_DIR + output_filename, "a") as file:
                file.write(json.dumps([row, col, val]) + "\n")

        if (val > 0.5):
            # x, y = point_to_movement(row, col, val)
            # l = np.linalg.norm([x, y])
            x_index = np.array(row)
            y_index = np.array(col)

            color_list = np.array(val)

            plt.xlim(0, 1)
            plt.ylim(0, 1)

            plt.scatter(x_index, y_index, c=color_list, marker="o")

        plt.pause(0.05)

    plt.ioff()

    plt.show()
    return


def main():
    # cali_each_point()
    scatter_plot()
    # draw_border()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', dest='config', action=make_action('store'),
                        default=None, help="specify configuration file")
    parser.add_argument('--mock', dest='mock', action=make_action('store'),
                        default=None, help="mock the input data")
    parser.add_argument('-o', dest='output', action=make_action('store'),
                        default=None, help="output filename")
    parser.add_argument('-f', dest='filename', action=make_action('store'),
                        default=None, help="filename read in")
    args = parser.parse_args()

    config = load_config(args.config)
    output_filename = args.output
    input_filename = args.filename

    with Uclient(
        udp=config['connection']['udp'],
        n=config['sensor']['shape']
    ) as tongue_client:
        my_processor = Processor(
            config['process']['interp'],
            blob=config['process']['blob'],
            threshold=config['process']['threshold'],
            order=config['process']['interp_order'],
            total=config['process']['blob_num'],
            special=config['process']['special_check'],
        )
        if (input_filename):
            my_generator = file_generator(input_filename)
        elif (args.mock):
            my_generator = mock_generator()
        else:
            my_generator = my_processor.gen_points(
                my_processor.gen_wrapper(tongue_client.gen()))
        main()
