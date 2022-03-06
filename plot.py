'''
    plot the pressure point on screen
'''

import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
from matsense.tools import load_config, make_action
from matsense.uclient import Uclient
from matsense.process import Processor

BASE_DATA_DIR = "./data/"


def mock_generator():
    while True:
        yield 0.5, 0.5, 1  # mock the data generated from processor


def file_generator(filename):
    with open(filename, "r") as f:
        data = f.read().split("\n")
    for line in data:
        line = json.loads(line)
        yield line[0], line[1], line[2]


def scatter_plot():
    '''
    scatter plot
    '''

    plt.ion()

    for _ in my_generator:
        # clean previous picture
        plt.cla()

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

            print(row, col, val)

            plt.xlim(0, 1)
            plt.ylim(0, 1)

            plt.scatter(x_index, y_index, c=color_list, marker="o")

        plt.pause(0.02)

    plt.ioff()

    plt.show()
    return


def main():
    scatter_plot()


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
            my_generator = my_processor.gen_points(tongue_client.gen())
        main()
