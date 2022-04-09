'''
    plot the pressure point on screen
'''
from socket import socket, AF_INET, SOCK_STREAM, timeout
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import random
import os
from multiprocessing import Process
try:
    import thread
except ImportError:
    import _thread as thread
from matsense.tools import load_config, make_action
from matsense.uclient import Uclient
from matsense.process import Processor
from gesture_typing import check_top_k, init_all, get_new_target_word

BASE_DATA_DIR = "./data/"
LETTER_IN_KEYBOARD = [['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', ''],
                      ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ''],
                      ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '']]
TYPE_IN_BREAK_TIME = 1  # seconds
CHOOSE_HOLD_TIME = 0.8  # seconds

# shared variable in threads
is_recording = False
gesture_typing_data = []
stop_showing = False
current_data = None
inst = None

IP = "localhost"
PORT = 8081


class CursorClient:
    def __init__(self, server_addr, port, timeout=1):
        self.my_socket = socket(AF_INET, SOCK_STREAM)
        self.my_socket.settimeout(timeout)
        self.connect(server_addr, port)

    def __exit__(self, type, value, traceback):
        self.close()

    def connect(self, address, port):
        self.my_socket.connect((address, port))
        print(f"client connecting to server: {address}")

    def close(self):
        self.my_socket.close()
        print("remote client socket closed")

    def sendToKB(self, touch_state, x, y):
        paras = [touch_state, x, y]
        self.my_socket.send(
            str(" ".join([str(item) for item in paras]) + "\n").encode())

    def sendToKBPlot(self, type, touch_state, x, y):
        # type should be in ['event', 'select', 'reshape']
        paras = [type, touch_state, x, y]
        self.my_socket.send(
            str(" ".join([str(item) for item in paras]) + "\n").encode())

    def selectWord(self, selectDirection):
        paras = ["select", selectDirection]
        self.my_socket.send(
            str(" ".join([str(item) for item in paras]) + "\n").encode())

    def reshapeKB(self, pos):
        # pos should be [0-1] * 12, as q_pos.x, q_pos.y, p_pos.x ...
        paras = ['reshape'] + pos
        self.my_socket.send(
            str(" ".join([str(item) for item in paras]) + "\n").encode())

    def setCandidates(self, cands):
        # len(cands) should >= 5
        paras = ['candidates'] + cands
        while (len(paras) < 6):
            paras.append(paras[-1])
        self.my_socket.send(
            str(" ".join([str(item) for item in paras]) + "\n").encode())

    def sendButton(self, cmd):
        self.my_socket.send((cmd+"\n").encode())

    def sendPressure(self, pressure):
        self.my_socket.send(('pressure' + " " + str(pressure) + "\n").encode())

    def sendMaxForce(self, pressure):
        self.my_socket.send(('maxforce' + " " + str(pressure) + "\n").encode())

    def sendPos(self, x, y):
        paras = ['grid', x, y]
        self.my_socket.send(
            str(" ".join([str(item) for item in paras]) + "\n").encode())
    
    def sendTargetWord(self, word):
        self.my_socket.send(('target' + " " + str(word) + "\n").encode())
    
    def sendRecordTimestamp(self, event):
        self.my_socket.send(('timestamp' + " " + str(event) + "\n").encode())

    def sendPressureRate(self, target, value):
        self.my_socket.send(('rate' + " " + str(target) + " " + str(value) + "\n").encode())
    
    def sendTimeLeft(self, time):
        self.my_socket.send(('timeleft' + " " + str(int(time)) + "\n").encode())


my_remote_handle = CursorClient(IP, PORT)


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
    col_3 = np.array([[[i * (0.75 - 0.25) / 7 + 0.25, 0],
                     [i * (0.75 - 0.25) / 7 + 0.25, 0.3]] for i in range(8)])
    col = [col_1, col_2, col_3]
    for line in row:
        plt.plot(line[:, 0], line[:, 1], color='#000000')
    gap_lengths = [(0.9 - 0.1) / 10, (0.8 - 0.2) / 9, (0.75 - 0.25) / 7]
    for i, c in enumerate(col):
        gap_length = gap_lengths[i]
        for j, line in enumerate(c):
            plt.plot(line[:, 0], line[:, 1], color='#000000')
            # 0.01 for half size of the letter
            plt.text(line[0][0] + gap_length / 2 - 0.01,
                     np.average([line[0][1], line[1][1]]
                                ), LETTER_IN_KEYBOARD[i][j],
                     fontsize=15)
    plt.show()


def scatter_plot(my_generator, output_filename):
    global is_recording, gesture_typing_data

    plt.ion()

    draw_border()

    prev_point = None

    for _ in my_generator:
        if stop_showing:
            break

        # generate data
        row, col, val = next(my_generator)

        # clean previous picture
        plt.cla()
        draw_border()

        # record all data
        if (output_filename != None):
            with open(BASE_DATA_DIR + output_filename, "a") as file:
                file.write(json.dumps([row, col, val]) + "\n")

        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # my_x_ticks = np.arange(0, 1, 0.1)
        # my_y_ticks = np.arange(0, 1, 0.1)
        # plt.xticks(my_x_ticks)
        # plt.yticks(my_y_ticks)

        if (val > 0.5):
            # if begin recording, then record data to gesture_typing_data
            # else clean it
            if (is_recording):
                gesture_typing_data.append([row, col, val])
            else:
                gesture_typing_data = []
            # x, y = point_to_movement(row, col, val)
            # l = np.linalg.norm([x, y])
            x_index = np.array(row)
            y_index = np.array(col)

            color_list = np.array(val)

            plt.scatter(x_index, y_index, c=color_list, marker="o", s=20)

            # if (prev_point != None):
            #     plt.plot([x_index, prev_point[0]], [y_index, prev_point[1]], '--', color="#000000", linewidth=2.5)
            prev_point = [x_index, y_index]

        plt.pause(0.05)

    plt.ioff()

    plt.show()
    return


def get_reshape_paras():
    global current_data, inst
    reshape_paras = []
    pos_name = ['q_pos', 'p_pos', 'a_pos', 'l_pos', 'z_pos', 'm_pos']
    pos = [np.array([0.5 * (0.9 - 0.1) / 10 + 0.1, 0.85]), np.array([-0.5 * (0.9 - 0.1) / 10 + 0.9, 0.85]), np.array([0.5 * (0.8 - 0.2) / 9 + 0.2, 0.5]),
           np.array([-0.5 * (0.8 - 0.2) / 9 + 0.8, 0.5]), np.array([0.5 * (0.75 - 0.25) / 9 + 0.25, 0.15]), np.array([-0.5 * (0.75 - 0.25) / 9 + 0.75, 0.15])]
    i = 0
    while (len(reshape_paras) < 6):
        input("please press " + pos_name[i])
        # (q_pos, p_pos, a_pos, l_pos, z_pos, m_pos) .x
        row = current_data[0]
        reshape_paras.append(max(row, pos[i][0]) if (
            i % 2 == 0) else min(row, pos[i][0]))
        print(row)
        i += 1
    inst = "idle"

    # for reshaping the keyboard
    my_remote_handle.reshapeKB([reshape_paras[0], 0.85,
                                reshape_paras[1], 0.85,
                                reshape_paras[2], 0.5,
                                reshape_paras[3], 0.5,
                                reshape_paras[4], 0.15,
                                reshape_paras[5], 0.15])
    return reshape_paras


def web_plot(my_generator, output_filename):
    global my_remote_handle, current_data, inst

    for _ in my_generator:
        if stop_showing:
            break

        # generate data
        row, col, val = next(my_generator)
        current_data = [row, col, val]

        # record all data
        if (output_filename != None):
            with open(BASE_DATA_DIR + output_filename, "a") as file:
                file.write(json.dumps([row, col, val, time.time()]) + "\n")
        
        if (val > 1):
            my_remote_handle.sendToKBPlot("event", 2, row, col)
            gesture_typing_data.append(current_data)
        else:
            my_remote_handle.sendToKBPlot("event", 2, -1, -1)

        time.sleep(0.01)


def grid_plot(my_generator, output_filename):
    for _ in my_generator:
        if stop_showing:
            break

        # generate data
        row, col, val = next(my_generator)
        current_data = [row, col, val]
        # record all data
        if (output_filename != None):
            with open(BASE_DATA_DIR + output_filename, "a") as file:
                file.write(json.dumps([row, col, val]) + "\n")

        if (val > 1):
            my_remote_handle.sendPos(row, col)
            gesture_typing_data.append(current_data)
        else:
            my_remote_handle.sendPos(-1, -1)

        time.sleep(0.015)

def cal_max_pressure(my_generator, continue_time):
    start_time = time.time()
    max_pressure = 0.0001
    print("Please use max pressure to press.")
    while (time.time() - start_time < continue_time):
        row, col, val = next(my_generator)
        if (val > max_pressure):
            max_pressure = val
    my_remote_handle.sendMaxForce(max_pressure)
    print("End max pressure test, begin the experiment.\nMax pressure:", max_pressure)
    return max_pressure


def pressure_plot(my_generator, output_filename, max_pressure):
    state = "idle"
    random_level = 0
    start_press_time = 0
    for _ in my_generator:
        if (state == "idle"):
            random_level = random.randint(0, 2)
            if (random_level == 0):
                my_remote_handle.sendPressureRate(50, 200)
                random_level = 0.25
            elif (random_level == 1):
                my_remote_handle.sendPressureRate(100, 200)
                random_level = 0.5
            else:
                my_remote_handle.sendPressureRate(150, 200)
                random_level = 0.75
            state = "wait"
        elif (state == "wait"):
            # generate data
            row, col, val = next(my_generator)
            my_remote_handle.sendPressure(val / max_pressure)
            # record all data
            if (output_filename != None):
                with open(BASE_DATA_DIR + output_filename, "a") as file:
                    file.write("wait " + json.dumps([row, col, val, time.time()]) + "\n")
            if (val > random_level * max_pressure):
                state = "press"
                start_press_time = time.time()
        elif (state == "press"):
            if (time.time() - start_press_time < 5):
                # generate data
                row, col, val = next(my_generator)
                my_remote_handle.sendPressure(val / max_pressure)
                my_remote_handle.sendTimeLeft(5 - (time.time() - start_press_time))
                # record all data
                if (output_filename != None):
                    with open(BASE_DATA_DIR + output_filename, "a") as file:
                        file.write("press " + json.dumps([row, col, val, time.time()]) + "\n")
            else:
                my_remote_handle.sendTimeLeft(0)
                row, col, val = next(my_generator)
                if (val < 0.05 * max_pressure):
                    state = "idle"
        time.sleep(0.015)


def interactive_mode(mode, interactive=False):
    global gesture_typing_data, stop_showing, my_remote_handle, current_data, inst, is_recording
    last_press_time = None
    chosen_word = None
    target_word = None
    while True:
        if (interactive):
            inst = input()
        if (inst == "r"):
            # recording
            is_recording = True  # only for scatter plot use
            row, col, val = current_data[0], current_data[1], current_data[2]
            if (val > 1):
                last_press_time = time.time()
            else:
                if (last_press_time):
                    if (time.time() - last_press_time > TYPE_IN_BREAK_TIME):
                        inst = "s"
                        last_press_time = None
        elif (inst == "s"):
            # stop recording and check top_k
            print("stop recording")
            if (len(gesture_typing_data) == 0):
                inst = "r"
            my_remote_handle.sendRecordTimestamp("stop")
            is_recording = False
            top_k = check_top_k(gesture_typing_data, 10)
            my_remote_handle.setCandidates([i for i in top_k[:5]])
            gesture_typing_data = []
            inst = "c"  # stop recording and then choose word at once
        elif (inst == "e"):
            stop_showing = True
            inst = None
        elif (inst == 'c'):
            # choose word
            start_time = time.time()
            start_status = [False, False]
            while (time.time() - start_time < CHOOSE_HOLD_TIME):
                if (current_data[2] < 1):
                    start_time = time.time()
                    continue
                if ((current_data[0] < 0.5) != start_status[0] or (current_data[1] < 0.5) != start_status[1]):
                    start_status[0] = current_data[0] < 0.5
                    start_status[1] = current_data[1] < 0.5
                    start_time = time.time()
            if (start_status[0]):
                if (start_status[1]):
                    my_remote_handle.selectWord('left')
                    chosen_word = top_k[3]
                else:
                    my_remote_handle.selectWord('up')
                    chosen_word = top_k[0]
            else:
                if (start_status[1]):
                    my_remote_handle.selectWord('down')
                    chosen_word = top_k[2]
                else:
                    my_remote_handle.selectWord('right')
                    chosen_word = top_k[1]
            print("chosen word", chosen_word)
            inst = 'idle'
        elif (inst == 'idle'):
            if (current_data[2] < 1):
                gesture_typing_data = []
                inst = 'r'
                if (mode == "training_keyboard"):
                    ## send target word
                    if target_word == None or chosen_word == None or chosen_word == target_word or try_times > 3:
                        target_word = get_new_target_word()
                        try_times = 0
                        my_remote_handle.sendTargetWord(target_word)
                    try_times += 1
                print("start typing")
                my_remote_handle.sendRecordTimestamp("start")

        time.sleep(0.1)


def main(my_generator, mode):
    # cali_each_point()
    # scatter_plot()
    # p = Process(target=web_plot, args=(my_generator, output_filename))
    # p.start()
    if (not os.path.exists(BASE_DATA_DIR)):
        os.makedirs(BASE_DATA_DIR)
    if (mode == "keyboard" or mode == "training_keyboard"):
        thread.start_new_thread(web_plot, (my_generator, output_filename))
        reshape_paras = get_reshape_paras()
        print("reshape_paras", reshape_paras)
        init_all(reshape_paras)
        interactive_mode(mode)
    elif (mode == "grid"):
        grid_plot(my_generator, output_filename)
    elif (mode == "pressure"):
        max_pressure = cal_max_pressure(my_generator, 5)
        pressure_plot(my_generator, output_filename, max_pressure)
    elif (mode == "draw"):
        web_plot(my_generator, output_filename)
        
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
        main(my_generator, config['application']['mode'])
