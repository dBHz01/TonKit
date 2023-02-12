from collections import deque
from socket import socket, AF_INET, SOCK_STREAM, timeout
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import random
import os
import pyautogui
import win32gui
from multiprocessing import Process
try:
    import thread
except ImportError:
    import _thread as thread
from matsense.tools import load_config, make_action, parse_mask
from matsense.uclient import Uclient
from matsense.process import Processor
from matsense.filemanager import write_line
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
current_data = None  # [row, col, val]
raw_data = None
inst = None

IP = "localhost"
# IP = "47.93.21.175"
PORT = 8081

pyautogui.PAUSE = 0.005


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
        self.my_socket.send(('rate' + " " + str(target) +
                            " " + str(value) + "\n").encode())

    def sendTimeLeft(self, time):
        self.my_socket.send(
            ('timeleft' + " " + str(int(time)) + "\n").encode())

    def sendCommand(self, paras):
        # paras should be a list of str
        self.my_socket.send(
            str(" ".join([str(item) for item in paras]) + "\n").encode())


my_remote_handle = CursorClient(IP, PORT)


def update_raw_data_wrapper(raw_generator, cali_array=np.array([])):
    global raw_data
    while True:
        raw_data = next(raw_generator)
        if (len(cali_array) > 0):
            raw_data[cali_array > 1] /= cali_array[cali_array > 1]
            raw_data[cali_array > 1] *= 10
        yield raw_data


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
    my_remote_handle.sendCommand(['reshape',
                                  reshape_paras[0], 0.85,
                                  reshape_paras[1], 0.85,
                                  reshape_paras[2], 0.5,
                                  reshape_paras[3], 0.5,
                                  reshape_paras[4], 0.15,
                                  reshape_paras[5], 0.15])
    return reshape_paras


def web_plot(my_generator, output_filename):
    global my_remote_handle, current_data, inst, raw_data

    for _ in my_generator:
        if stop_showing:
            break

        # generate data
        row, col, val = next(my_generator)
        current_data = [row, col, val]
        # print(val)

        # record all data
        if (output_filename != None):
            write_line(BASE_DATA_DIR + output_filename,
                       raw_data.reshape(-1), [row, col, val, time.time()])

        if (val > 1):
            my_remote_handle.sendCommand(["event", 2, row, col])
            gesture_typing_data.append(current_data)
        else:
            my_remote_handle.sendCommand(["event", 2, -1, -1])

        time.sleep(0.01)


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
            my_remote_handle.sendCommand(["status", "choose"])
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
                    my_remote_handle.sendCommand(["select", 'left'])
                    chosen_word = top_k[3]
                else:
                    my_remote_handle.sendCommand(["select", 'up'])
                    chosen_word = top_k[0]
            else:
                if (start_status[1]):
                    my_remote_handle.sendCommand(["select", 'down'])
                    chosen_word = top_k[2]
                else:
                    my_remote_handle.sendCommand(["select", 'right'])
                    chosen_word = top_k[1]
            print("chosen word", chosen_word)
            inst = 'idle'
            my_remote_handle.sendCommand(["status", "wait"])
        elif (inst == 'idle'):
            if (current_data[2] < 1):
                gesture_typing_data = []
                inst = 'r'
                if (mode == "training_keyboard"):
                    # send target word
                    if target_word == None or chosen_word == None or chosen_word == target_word or try_times > 3:
                        target_word = get_new_target_word()
                        try_times = 0
                        my_remote_handle.sendTargetWord(target_word)
                    try_times += 1
                print("start typing")
                my_remote_handle.sendRecordTimestamp("start")
                my_remote_handle.sendCommand(["status", "type"])

        time.sleep(0.1)


def move_mouse(output_filename):
    global raw_data, gesture_typing_data, stop_showing, my_remote_handle, current_data, inst, is_recording
    control_type = "mouse"
    MIN_VAL = 1
    MOVE_WIN_SIZE = 5
    MIN_MOVE_LEN = 0.06
    last_movements = deque(maxlen=MOVE_WIN_SIZE)
    last_point = np.array([])
    movement = np.array([0, 0], dtype=float)
    min_x, min_y, max_x, max_y = [10000, 10000, -1, -1]
    last_press_time = None
    chosen_word = None
    change_to_mouse = False
    while True:
        if control_type == "mouse":
            start_time = time.time()
            # generate data
            row, col, val = current_data[0], current_data[1], current_data[2]
            cur_point = np.array([row, col], dtype=float)
            # record all data
            if (output_filename != None):
                write_line(BASE_DATA_DIR + output_filename,
                        raw_data.reshape(-1), [row, col, val, time.time()])
            if val > MIN_VAL:
                min_x = min(min_x, row)
                min_y = min(min_y, col)
                max_x = max(max_x, row)
                max_y = max(max_y, col)
                if len(last_point) != 0:
                    cur_movement = cur_point - last_point
                    # print(cur_movement)
                    if len(last_movements) < MOVE_WIN_SIZE:
                        if abs(cur_movement[0]) < MIN_MOVE_LEN and abs(cur_movement[1]) < MIN_MOVE_LEN:
                            last_movements.append(cur_movement)
                            movement += (cur_movement) / 5
                    else:
                        if abs(cur_movement[0]) < MIN_MOVE_LEN and abs(cur_movement[1]) < MIN_MOVE_LEN:
                            del_movement = last_movements.popleft()
                            last_movements.append(cur_movement)
                            movement -= del_movement / 5
                            movement += (cur_movement) / 5
                            pyautogui.move(
                                movement[0] * 1000, -1 * movement[1] * 1000)
                            # print(movement * 1000)
                        # print([np.linalg.norm(i) for i in last_movements])
                last_point = cur_point
            else:
                if [min_x, min_y, max_x, max_y] != [10000, 10000, -1, -1]:
                #     print(min_x, max_x, min_y, max_y)
                #     print(max_x - min_x, max_y - min_y)
                    if max_x - min_x < 0.1 and max_y - min_y < 0.1:
                        pyautogui.click()
                        if win32gui.GetCursorInfo()[1] == 65541:
                            control_type = "keyboard"
                            inst = "idle"
                min_x, min_y, max_x, max_y = [10000, 10000, -1, -1]
                
            time.sleep(0.002)
            end_time = time.time()
            # print("fps: ", str(1 / (end_time - start_time)))
        elif control_type == "keyboard":
            if (inst == "r"):
                # recording
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
                my_remote_handle.sendCommand(["status", "choose"])
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
                        my_remote_handle.sendCommand(["select", 'left'])
                        chosen_word = top_k[3]
                    else:
                        my_remote_handle.sendCommand(["select", 'up'])
                        chosen_word = top_k[0]
                else:
                    if (start_status[1]):
                        my_remote_handle.sendCommand(["select", 'down'])
                        chosen_word = top_k[2]
                    else:
                        my_remote_handle.sendCommand(["select", 'right'])
                        chosen_word = top_k[1]
                print("chosen word", chosen_word)
                if chosen_word == "qzmp":
                    change_to_mouse = True
                else:
                    pyautogui.write(chosen_word + " ", interval=0.02)
                inst = 'idle'
                my_remote_handle.sendCommand(["status", "wait"])
            elif (inst == 'idle'):
                if (current_data[2] < 1):
                    if (change_to_mouse):
                        control_type = "mouse"
                        continue
                    gesture_typing_data = []
                    inst = 'r'
                    print("start typing")
                    my_remote_handle.sendRecordTimestamp("start")
                    my_remote_handle.sendCommand(["status", "type"])

            time.sleep(0.1)


def main(my_generator):
    if (not os.path.exists(BASE_DATA_DIR)):
        os.makedirs(BASE_DATA_DIR)
    # if (mode == "keyboard" or mode == "training_keyboard"):
    #     thread.start_new_thread(web_plot, (my_generator, output_filename))
    #     reshape_paras = get_reshape_paras()
    #     print("reshape_paras", reshape_paras)
    #     init_all(reshape_paras)
    #     interactive_mode(mode)
    # elif (mode == "move_mouse"):
    thread.start_new_thread(web_plot, (my_generator, output_filename))
    reshape_paras = get_reshape_paras()
    print("reshape_paras", reshape_paras)
    init_all(reshape_paras)
    time.sleep(1)
    move_mouse(output_filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', dest='config', action=make_action('store'),
                        default=None, help="specify configuration file")
    parser.add_argument('-o', dest='output', action=make_action('store'),
                        default=None, help="output filename")
    args = parser.parse_args()

    config = load_config(args.config)
    output_filename = None
    if 'application' in config and 'filename' in config['application']:
        output_filename = config['application']['filename']
    if (args.output):
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
        my_generator = my_processor.gen_points(
                update_raw_data_wrapper(
                    my_processor.gen_wrapper(tongue_client.gen())))
        main(my_generator)
