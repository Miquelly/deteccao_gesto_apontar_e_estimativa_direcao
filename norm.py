import numpy as np
import matplotlib

matplotlib.use("Agg")
from is_msgs.image_pb2 import HumanKeypoints as HKP
from math import cos, sin

# from Classificador import Gesture
### Setting printing options
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
np.set_printoptions(precision=3, suppress=True)

TO_COCO_IDX = {
    HKP.Value("NOSE"): 0,
    HKP.Value("LEFT_EYE"): 1,
    HKP.Value("RIGHT_EYE"): 2,
    HKP.Value("LEFT_EAR"): 3,
    HKP.Value("RIGHT_EAR"): 4,
    HKP.Value("LEFT_SHOULDER"): 5,
    HKP.Value("RIGHT_SHOULDER"): 6,
    HKP.Value("LEFT_ELBOW"): 7,
    HKP.Value("RIGHT_ELBOW"): 8,
    HKP.Value("LEFT_WRIST"): 9,
    HKP.Value("RIGHT_WRIST"): 10,
    HKP.Value("LEFT_HIP"): 11,
    HKP.Value("RIGHT_HIP"): 12,
    HKP.Value("LEFT_KNEE"): 13,
    HKP.Value("RIGHT_KNEE"): 14,
    HKP.Value("LEFT_ANKLE"): 15,
    HKP.Value("RIGHT_ANKLE"): 16,
}


def z_rotation(angle):
    rotation_matrix = np.array(
        [
            [cos(angle), -sin(angle), 0, 0],
            [sin(angle), cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return rotation_matrix


def move(dx, dy, dz):
    T = np.eye(4)
    T[0, -1] = dx
    T[1, -1] = dy
    T[2, -1] = dz
    return T


def normalizacao(sk):

    skNorm = np.transpose(sk)
    skNorm = np.vstack([skNorm, np.ones(np.size(skNorm, 1))])

    dx = -sk[TO_COCO_IDX[HKP.Value("NOSE")]][0]
    dy = -sk[TO_COCO_IDX[HKP.Value("NOSE")]][1]
    dz = 0

    T = move(dx, dy, dz)

    # print("T: ", T)

    skNorm = np.transpose(sk)
    num_columns = np.size(skNorm, 1)
    ones_line = np.ones(num_columns)
    skNorm = np.vstack([skNorm, ones_line])

    # print("skNorm: ", skNorm)

    skT = np.dot(T, skNorm)
    skT = np.transpose(skT).reshape(17, 4, 1)

    # print("skT: ", skT[0:2])
    p0 = [[0], [0], skT[TO_COCO_IDX[HKP.Value("RIGHT_SHOULDER")]][2], [1.0]]
    p1 = [[1], [0], skT[TO_COCO_IDX[HKP.Value("RIGHT_SHOULDER")]][2], [1.0]]
    p2 = skT[TO_COCO_IDX[HKP.Value("RIGHT_SHOULDER")]]

    direcao = skT[TO_COCO_IDX[HKP.Value("RIGHT_SHOULDER")]][1] / np.abs(
        skT[TO_COCO_IDX[HKP.Value("RIGHT_SHOULDER")]][1]
    )

    # print("p0: ", p0, "p2: ", p2[0], "p2: ", p2, "p3: ", p3)

    vetor1 = np.subtract(p1, p0).reshape(1, 4)[0][0:3]
    vetor2 = np.subtract(p2, p0).reshape(1, 4)[0][0:3]

    norm_vetor1 = np.linalg.norm(vetor1)
    norm_vetor2 = np.linalg.norm(vetor2)

    prod_interno = np.dot(vetor1, vetor2)

    if norm_vetor1 != 0 and norm_vetor2 != 0:
        sim = prod_interno / (norm_vetor1 * norm_vetor2)
    else:
        sim = 0

    # angle = (180 * np.arccos(sim)) / np.pi

    if direcao < 0:
        angle = np.arccos(sim)
    else:
        angle = -np.arccos(sim)

    # print("Angle: ", angle)

    Rz = z_rotation(angle)

    # print("Rz: ", Rz)

    M = Rz @ T

    # print("M: ", M)

    skM = np.dot(M, skNorm)
    skM = np.transpose(skM).reshape(17, 4, 1)

    # print("skM: ", skM[0:2])

    skM = skM.reshape(17, 4)[:, 0:3]

    # print("skM: ", skM)

    return skM
