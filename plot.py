'''
    plot the pressure point on screen
'''

import argparse
import matplotlib.pyplot as plt
import numpy as np
from matsense.tools import load_config, make_action
from matsense.uclient import Uclient
from matsense.process import Processor


# def point_generator(generator):
#     row, col, val = next(generator)


# def pressure_function(val):
#     if (val < 1):
#         return val
#     elif (val > 6):
#         return 36 + 0.5 * val
#     else:
#         return val ** 2


# def distance_function(dist):
#     if (dist <= 0.2):
#         return dist
#     elif (dist <= 0.3):
#         return dist / 0.3
#     else:
#         return 1


# def point_to_movement(row, col, val):
#     """transform a pressure point in (0,1) (0,1) square to actual move point

#     Args:
#         row (float): 0 to 1
#         col (float): 0 to 1
#         val (float): pressure value


#     Returns:
#         x, y value of movement
#     """
#     direction = np.array([row - 0.5, col - 0.5])
#     distance_to_center = np.linalg.norm(direction)
#     direction = direction / distance_to_center
#     x = direction[0] * pressure_function(val) * \
#         distance_function(distance_to_center) * 1.5
#     y = direction[1] * \
#         pressure_function(val) * distance_function(distance_to_center)
#     return x, y

def mock_generator():
    while True:
        yield 0.5, 0.5, 1  # mock the data generated from processor


def scatter_plot():
    '''
    scatter plot
    '''

    plt.ion()

    # 循环
    for _ in my_generator:
        # clean previous picture
        plt.cla()

        row, col, val = next(my_generator)
        if (val > 0.5):
            # x, y = point_to_movement(row, col, val)
            # l = np.linalg.norm([x, y])
            x_index = np.array(row)
            y_index = np.array(col)

            # 设置相关参数
            color_list = np.array(val)

            print(row, col, val)

            plt.xlim(0, 1)
            plt.ylim(0, 1)

            # 画散点图
            plt.scatter(x_index, y_index, c=color_list, marker="o")

        # 暂停
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
    parser.add_argument('-o', dest='output', action=make_action('store'),
                        default=None, help="output filename")
    args = parser.parse_args()
    config = load_config(args.config)
    output_filename = args.output
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
        # my_generator = my_processor.gen_points(tongue_client.gen())
        my_generator = mock_generator()
        main()
