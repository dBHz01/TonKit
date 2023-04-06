from socket import socket, AF_INET, SOCK_STREAM, timeout

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