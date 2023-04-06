from collections import deque
import argparse
import numpy as np
import json
import time
import os
import pyautogui
from matsense.tools import load_config, make_action, parse_mask
from matsense.uclient import Uclient
from matsense.process import Processor
from matsense.filemanager import write_line
from one_euro_filter import OneEuroFilter

BASE_DATA_DIR = "./data/"

# shared variable in threads
raw_data = None

pyautogui.PAUSE = 0.005


def mock_generator():
    while True:
        yield 0.5, 0.5, 1  # mock the data generated from processor


def file_generator(filename):
    with open(filename, "r") as f:
        data = f.read().split("\n")
    for line in data:
        line = json.loads(line)
        yield line[0], line[1], line[2]


def update_raw_data_wrapper(raw_generator, cali_array=np.array([])):
    global raw_data
    while True:
        raw_data = next(raw_generator)
        if len(cali_array) > 0:
            raw_data[cali_array > 1] /= cali_array[cali_array > 1]
            raw_data[cali_array > 1] *= 10
        yield raw_data


def check_click_area(cur_point):
    '''
    check if the current point is in the click area
    '''
    print(cur_point)
    click_area = np.array([0, 0.8, 1, 0.2]) # [x, y, w, h]
    if (
        cur_point[0] > click_area[0]
        and cur_point[0] < click_area[0] + click_area[2]
        and cur_point[1] > click_area[1]
        and cur_point[1] < click_area[1] + click_area[3]
    ):
        return True
    else:
        return False


def move_cursor(my_generator, output_filename):
    global raw_data
    MIN_VAL = 2
    MOVE_WIN_SIZE = 5
    MAX_MOVE_LEN = 0.06
    LONG_PRESS_TIME = 0.6
    last_movements = deque(maxlen=MOVE_WIN_SIZE + 1)
    last_point = np.array([])
    movement = np.array([0, 0], dtype=float)
    one_euro_filter_x = None
    one_euro_filter_y = None
    in_click_area_start_time = 0 # 0 means no data, -1 means out of click area, >0 means in click area
    is_long_pressing = False
    while True:
        start_time = time.time()

        # generate data
        row, col, val = next(my_generator)
        cur_point = np.array([row, col], dtype=float)

        # record all data
        if output_filename != None:
            write_line(
                BASE_DATA_DIR + output_filename,
                raw_data.reshape(-1),
                [row, col, val, time.time()],
            )

        if val > MIN_VAL:  # remove low value
            # print(val)
            if len(last_point) == 0:
                # init filter
                one_euro_filter_x = OneEuroFilter(time.time(), row, beta=0.006)
                one_euro_filter_y = OneEuroFilter(time.time(), col, beta=0.006)
            else:
                # get filtered data
                cur_point = np.array(
                    [
                        one_euro_filter_x(time.time(), row),
                        one_euro_filter_y(time.time(), col),
                    ],
                    dtype=float,
                )
                # get raw data
                # cur_point = np.array([row, col], dtype=float)

                if in_click_area_start_time >= 0:
                    if check_click_area(cur_point):
                        if in_click_area_start_time == 0:
                            in_click_area_start_time = time.time()
                    else:
                        # out of click area
                        in_click_area_start_time = -1
                else:
                    cur_movement = cur_point - last_point

                    # move cursor with moving average
                    if (
                        abs(cur_movement[0]) < MAX_MOVE_LEN
                        and abs(cur_movement[1]) < MAX_MOVE_LEN
                    ):
                        last_movements.append(cur_movement)
                        movement += cur_movement / 5
                        if len(last_movements) > MOVE_WIN_SIZE:
                            del_movement = last_movements.popleft()
                            movement -= del_movement / 5
                            pyautogui.move(movement[0] * 1000, -1 * movement[1] * 1000)
            last_point = cur_point
        elif len(last_point) > 0:
            # end of movement
            if in_click_area_start_time > 0:
                print(time.time() - in_click_area_start_time)
                if time.time() - in_click_area_start_time > LONG_PRESS_TIME:
                    print("long press")
                    pyautogui.mouseDown()
                    is_long_pressing = True
                else:
                    if is_long_pressing:
                        pyautogui.mouseUp()
                        is_long_pressing = False
                    else:
                        print("click")
                        pyautogui.click()
            in_click_area_start_time = 0
                

        time.sleep(0.002)
        end_time = time.time()
        # print("fps: ", str(1 / (end_time - start_time)))


def main(my_generator, mode, output_filename=None):
    if not os.path.exists(BASE_DATA_DIR):
        os.makedirs(BASE_DATA_DIR)
    move_cursor(my_generator, output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        dest="config",
        action=make_action("store"),
        default=None,
        help="specify configuration file",
    )
    parser.add_argument(
        "--mock",
        dest="mock",
        action=make_action("store"),
        default=None,
        help="mock the input data",
    )
    parser.add_argument(
        "-o",
        dest="output",
        action=make_action("store"),
        default=None,
        help="output filename",
    )
    parser.add_argument(
        "-f",
        dest="filename",
        action=make_action("store"),
        default=None,
        help="filename read in",
    )
    parser.add_argument(
        "-c",
        dest="cali_filename",
        action=make_action("store"),
        default=None,
        help="calibrate filename",
    )
    parser.add_argument(
        "-s",
        dest="skip_reshape",
        action=make_action("store"),
        default=None,
        help="skip reshape",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_filename = None
    if "application" in config and "filename" in config["application"]:
        output_filename = config["application"]["filename"]
    if args.output:
        output_filename = args.output
    input_filename = args.filename
    cali_filename = args.cali_filename
    if cali_filename:
        with open(cali_filename, "r", encoding="UTF-8") as f:
            cali_array = parse_mask(f.read())
    else:
        cali_array = None

    with Uclient(
        udp=config["connection"]["udp"], n=config["sensor"]["shape"]
    ) as tongue_client:
        my_processor = Processor(
            config["process"]["interp"],
            blob=config["process"]["blob"],
            threshold=config["process"]["threshold"],
            order=config["process"]["interp_order"],
            total=config["process"]["blob_num"],
            special=config["process"]["special_check"],
        )
        if input_filename:
            my_generator = file_generator(input_filename)
        elif args.mock:
            my_generator = mock_generator()
        elif cali_array:
            my_generator = my_processor.gen_points(
                update_raw_data_wrapper(
                    my_processor.gen_wrapper(tongue_client.gen()), cali_array
                )
            )
        else:
            my_generator = my_processor.gen_points(
                update_raw_data_wrapper(my_processor.gen_wrapper(tongue_client.gen()))
            )
        main(
            my_generator, config["application"]["mode"], output_filename=output_filename
        )
