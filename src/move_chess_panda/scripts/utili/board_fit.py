import numpy as np
import math
from math import sin
from math import cos


def A1(s, x0, y0, th):
    x = x0
    y = y0

    return [x, y]


def H1(s, x0, y0, th):
    x = x0 + s * cos(th)
    y = y0 + s * sin(th)

    return [x, y]


def H8(s, x0, y0, th):
    x = x0 + s * cos(th) - s * sin(th)
    y = y0 + s * sin(th) + s * cos(th)

    return [x, y]


def A8(s, x0, y0, th):
    x = x0 - s * sin(th)
    y = y0 + s * cos(th)

    return [x, y]


def board(s, x, y, th, h):
    board = np.zeros((4, 3))
    board[0, :-1] = A1(s, x, y, th)
    board[1, :-1] = H1(s, x, y, th)
    board[2, :-1] = H8(s, x, y, th)
    board[3, :-1] = A8(s, x, y, th)

    board[:, 2] = h

    return board


def distance(var, x, y, th):
    markers = var["markers"]
    s = var["size"]
    a1 = A1(s, x, y, th)
    h1 = H1(s, x, y, th)
    h8 = H8(s, x, y, th)
    a8 = A8(s, x, y, th)

    distance = []

    for m in range(len(markers)):
        dist_a1 = math.sqrt((a1[0] - markers[m, 0]) ** 2 + (a1[1] - markers[m, 1]) ** 2)
        dist_h1 = math.sqrt((h1[0] - markers[m, 0]) ** 2 + (h1[1] - markers[m, 1]) ** 2)
        dist_h8 = math.sqrt((h8[0] - markers[m, 0]) ** 2 + (h8[1] - markers[m, 1]) ** 2)
        dist_a8 = math.sqrt((a8[0] - markers[m, 0]) ** 2 + (a8[1] - markers[m, 1]) ** 2)

        distance.append(min(dist_a1, dist_h1, dist_h8, dist_a8))

    return distance
