import numpy as np
from math import sqrt
import random
import matplotlib.pyplot as plt
from queue import PriorityQueue
from dtaidistance import dtw_ndim

LETTER_IN_KEYBOARD = [['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
                      ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
                      ['Z', 'X', 'C', 'V', 'B', 'N', 'M']]
WORD_SET = set()
WORD_LIST = list()
DIST_PATTERN = {}
STD_KB_POS = {}
WORD_PROB = {}
WORD_NUM = 3500
WORD_DICT_NAME = "./data/words_10000.txt"


def euclidean_distance(p: np.ndarray, q: np.ndarray) -> float:
    return sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)


def init_all(reshape_paras):
    init_word_set(WORD_NUM)
    init_keyboard_pos(reshape_paras)
    init_pattern_set()


def init_word_set(num):
    global WORD_PROB, WORD_LIST
    WORD_SET.add("qzmp")
    WORD_PROB["qzmp"] = 3.659327655609967e-05
    with open(WORD_DICT_NAME, 'r') as file:
        if (WORD_DICT_NAME == "./data/word_5000.csv"):
            lines = file.readlines()
            for i in range(len(lines)):
                word = lines[i].lower().split(' ')[0]
                if (word in WORD_SET):
                    continue
                if (num == 0):
                    break
                prob = float(lines[i].lower().split(' ')[2])
                WORD_SET.add(word)
                WORD_PROB[word] = prob
                num -= 1
        elif (WORD_DICT_NAME == "./data/words_10000.txt"):
            lines = file.readlines()
            for i in range(len(lines)):
                word = lines[i].lower().replace('\n', '').split('\t')[0]
                if (word in WORD_SET):
                    continue
                if (num == 0):
                    break
                prob = float(lines[i].lower().replace('\n', '').split('\t')[2])
                WORD_SET.add(word)
                WORD_PROB[word] = prob
                num -= 1
    WORD_LIST = list(WORD_SET)


def generate_standard_pattern(word: str, num_nodes: int, distribute):
    word = word.upper()
    nodes = []
    pattern = []
    for i, c in enumerate(word):
        if (i > 0 and word[i] == word[i - 1]):
            continue
        nodes.append(STD_KB_POS[c])
    if len(nodes) == 1:
        return [nodes[0] for i in range(num_nodes)]
    total_len = 0
    for i in range(len(nodes) - 1):
        total_len += euclidean_distance(nodes[i], nodes[i + 1])
    num_pieces = num_nodes - 1
    used_pieces = 0
    for i in range(len(nodes) - 1):
        if i == len(nodes) - 2:
            p1 = num_pieces - used_pieces
        else:
            d1 = euclidean_distance(nodes[i], nodes[i + 1])
            p1 = int(d1 * num_pieces / total_len)
        if p1 == 0:
            continue
        delta_x = nodes[i + 1][0] - nodes[i][0]
        delta_y = nodes[i + 1][1] - nodes[i][1]
        for j in range(0, p1):
            pattern.append(
                np.array([
                    nodes[i][0] + delta_x * distribute(j / p1),
                    nodes[i][1] + delta_y * distribute(j / p1)
                ]))
        used_pieces += p1
    pattern.append(nodes[-1])
    if len(pattern) != num_nodes:
        print(word, num_nodes, len(pattern))
        raise Exception()
    return np.array(pattern)


def init_pattern_set():
    global DIST_PATTERN
    for w in WORD_SET:
        w_with_q = w
        # if (w[0].lower() != 'q'):
        #     w_with_q = "q" + w
        dist_path = generate_standard_pattern(
            w_with_q, int((len(w) * 6.6457 + 4.2080) / 2),
            lambda x: x)
        # dist_path = generate_standard_pattern(
        #     w_with_q, int((len(w) * 6.6457 + 4.2080) / 2),
        #     lambda x: -2 * x**3 + 3 * x**2)
        dist_path_x = [d[0] for d in dist_path]
        dist_path_y = [d[1] for d in dist_path]
        DIST_PATTERN[w] = list(zip(dist_path_x, dist_path_y))


def init_keyboard_pos(reshape_paras):
    '''
    This function generates linear keyboard position
    '''
    global STD_KB_POS
    # q_pos = np.array([0.5 * (0.9 - 0.1) / 10 + 0.1, 0.85])
    # p_pos = np.array([-0.5 * (0.9 - 0.1) / 10 + 0.9, 0.85])
    # a_pos = np.array([0.5 * (0.8 - 0.2) / 9 + 0.2, 0.5])
    # l_pos = np.array([-0.5 * (0.8 - 0.2) / 9 + 0.8, 0.5])
    # z_pos = np.array([0.5 * (0.75 - 0.25) / 9 + 0.25, 0.15])
    # m_pos = np.array([-0.5 * (0.75 - 0.25) / 9 + 0.75, 0.15])
    q_pos = np.array([reshape_paras[0], 0.85])
    p_pos = np.array([reshape_paras[1], 0.85])
    a_pos = np.array([reshape_paras[2], 0.5])
    l_pos = np.array([reshape_paras[3], 0.5])
    z_pos = np.array([reshape_paras[4], 0.15])
    m_pos = np.array([reshape_paras[5], 0.15])
    pos = {}
    pos.update({letter: ((9 - i) * q_pos + i * p_pos) / 9
               for i, letter in enumerate(LETTER_IN_KEYBOARD[0])})
    pos.update({letter: ((8 - i) * a_pos + i * l_pos) / 8
               for i, letter in enumerate(LETTER_IN_KEYBOARD[1])})
    pos.update({letter: ((6 - i) * z_pos + i * m_pos) / 6
               for i, letter in enumerate(LETTER_IN_KEYBOARD[2])})
    STD_KB_POS = pos


def get_new_target_word():
    return WORD_LIST[random.randint(0, WORD_NUM - 1)]


def resample_path(path):
    # print("path", path)
    sampleSize = 150
    n = len(path)
    ret = []
    if (n == 1):
        for i in range(sampleSize):
            ret.append(list(path[0]))
        return ret
    length = 0
    for i in range(n-1):
        length += euclidean_distance(np.array(path[i]), np.array(path[i + 1]))
    interval = length / (sampleSize - 1)
    lastPos = path[0]
    currLen = 0
    no = 1
    ret.append(list(path[0]))
    while (no < n):
        dist = euclidean_distance(np.array(lastPos), np.array(path[no]))
        if (currLen + dist >= interval and dist > 0):
            ratio = (interval - currLen) / dist
            tmpPos = lastPos.copy()
            lastPos = [
                tmpPos[0] + ratio * (path[no][0] - tmpPos[0]),
                tmpPos[1] + ratio * (path[no][1] - tmpPos[1]),
            ]
            ret.append(lastPos)
            currLen = 0
        else:
            currLen += dist
            lastPos = list(path[no])
            no += 1
    for i in range(len(ret), sampleSize):
        ret.append(list(path[n - 1]))
    ret = np.array(ret)
    # print("ret", ret)
    return ret


def check_top_k(data, k):
    '''
    data = [[x, y, z], ...]
    '''
    data = np.array(data)
    # print(data)
    total = 0
    # top_k = [0] * k
    prev = None
    if data is None:
        return
    user_path_x = data[:, 0]
    user_path_y = data[:, 1]
    # depths = data[:, 2]
    # user = genVectors(x, y, depths)
    # word = get_word(i, j)
    total += 1
    # x, y = downsample(x, y, depths, 2)
    user_path = np.array(list(zip(user_path_x, user_path_y)))
    user_path = resample_path(user_path)
    q = PriorityQueue()

    for w in WORD_SET:
        # theta distance
        # pattern = THETA_PATTERN[w]
        # if (len(user) <= 0 or len(pattern) <= 0):
        #     continue
        # d1, cost_matrix, acc_cost_matrix, path = a_dtw(
        #     user, pattern, dist=distance)

        # shape distance
        pattern = DIST_PATTERN[w]
        if (len(user_path) <= 0 or len(pattern) <= 0):
            continue
        d2 = dtw_ndim.distance_fast(user_path, np.array(
            pattern), use_pruning=True, window=8)

        if (w in WORD_PROB):
            d1 = np.log10(WORD_PROB[w])

        # # bigram probabilities
        # if prev in BIWORD_PROB and w in BIWORD_PROB[prev]:
        #     d = BIWORD_PROB[prev][w]
        # elif w in WORD_PROB:
        #     d = WORD_PROB[w]
        # else:
        #     d = 10**(-10)

        # q.put((0.6 * d2 - 0.4 * 0.1 * np.log10(d), w))
        q.put((-1 * 0.2 * 0.5 * d1 + 0.8 * d2, w))
        # q.put((d2, w))

    top = []
    top_acc = []
    for p in range(k):
        try:
            tmp = q.get_nowait()
            top.append(tmp[1])
            top_acc.append(tmp[0])
            # if word in top:
            #     top_k[p] += 1
        except:
            break
    # print(word, top[:10])
    print(top[:k])
    print(top_acc[:k])
    # prev = word
    # print("total: %d" % total)
    # for i in range(k):
    #     print("top%d acc: %f" % (i + 1, top_k[i] / total))
    return top[:k]


if __name__ == "__main__":
    init_all([0.5 * (0.9 - 0.1) / 10 + 0.1, -0.5 * (0.9 - 0.1) / 10 + 0.9, 0.5 * (0.8 - 0.2) / 9 + 0.2,
              -0.5 * (0.8 - 0.2) / 9 + 0.8, 0.5 * (0.75 - 0.25) / 9 + 0.25, -0.5 * (0.75 - 0.25) / 9 + 0.75])
    for i in WORD_SET:
        print(i)
        check_top_k(DIST_PATTERN[i], 4)
