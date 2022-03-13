import numpy as np

LETTER_IN_KEYBOARD = [['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
                      ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
                      ['Z', 'X', 'C', 'V', 'B', 'N', 'M']]


def gen_keyboard_pos():
    '''
    This function generates linear keyboard position
    '''
    q_pos = np.array([0.5 * (0.9 - 0.1) / 10 + 0.1, 0.85])
    p_pos = np.array([-0.5 * (0.9 - 0.1) / 10 + 0.9, 0.85])
    a_pos = np.array([0.5 * (0.8 - 0.2) / 9 + 0.2, 0.5])
    l_pos = np.array([-0.5 * (0.8 - 0.2) / 9 + 0.8, 0.5])
    z_pos = np.array([0.5 * (0.65 - 0.35) / 9 + 0.35, 0.15])
    m_pos = np.array([-0.5 * (0.65 - 0.35) / 9 + 0.65, 0.15])
    pos = {}
    pos.update({letter: ((9 - i) * q_pos + i * p_pos) / 9
               for i, letter in enumerate(LETTER_IN_KEYBOARD[0])})
    pos.update({letter: ((8 - i) * a_pos + i * l_pos) / 8
               for i, letter in enumerate(LETTER_IN_KEYBOARD[1])})
    pos.update({letter: ((6 - i) * z_pos + i * m_pos) / 6
               for i, letter in enumerate(LETTER_IN_KEYBOARD[2])})
    return pos


if __name__ == "__main__":
    gen_keyboard_pos()