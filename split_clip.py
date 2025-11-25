import json
import os

import numpy as np
from google.protobuf.json_format import ParseDict
from is_msgs.image_pb2 import HumanKeypoints, ObjectAnnotations
from typing import Dict, Tuple, List


def split_clip(person) -> List[Tuple[int, int]]:
    spots = []
    ifes = {
        "folder": "/home/miquelly/Documents/ifes",
        "person_ids": [
            person,
            # 1,
            # 2,
            # 3,
            # 4,
            # 5,
            # 6,
            # 7,
            # 8,
            # 9,
            # 10,
            # 11,
            # 12,
            # 13,
            # 14,
            # 15,
            # 16,
            # 17,
            # 18,
        ],
        "gesture_ids": [
            # 1,
            # 2,
            # 3,
            # 4,
            # 5,
            # 6,
            # 7,
            # 8,
            # 9,
            # 10,
            11,
            # 12,
            # 13,
            # 14,
            # 15,
        ],
    }

    ufes = {
        "folder": "/home/miquelly/Desktop/IC2/Segundo/ufes",
        "person_ids": [
            person,
            # 1,
            # 2,
        ],
        "gesture_ids": [
            # 1,
            # 2,
            # 3,
            # 4,
            # 5,
            # 6,
            # 7,
            # 8,
            # 9,
            # 10,
            11,
            # 12,
            # 13,
            # 14,
            # 15,
        ],
    }

    banco_dados = {
        "folder": "/home/miquelly/Documents/banco_dados",
        "person_ids": [
            person,
            # 1,
            # 2,
            # 3,
            # 4,
            # 5,
            # 6,
            # 7,
            # 8,
            # 9,
            # 10,
            # 11,
            # 12,
            # 13,
            # 14,
            # 15,
            # 16,
            # 17,
            # 18,
        ],
        "gesture_ids": [
            # 1,
            # 2,
            # 3,
            # 4,
            # 5,
            # 6,
            # 7,
            # 8,
            # 9,
            # 10,
            11,
            # 12,
            # 13,
            # 14,
            # 15,
        ],
    }
    setup = banco_dados

    for person_id in setup["person_ids"]:
        for gesture_id in setup["gesture_ids"]:

            filename_path = os.path.join(
                setup["folder"],
                "p{:03d}g{:02d}_spots.json".format(person_id, gesture_id),
            )
            with open(filename_path) as json_file:
                spots_data = json.load(json_file)

            filename_path = os.path.join(
                "./banco_dados",
                "p{:03d}g{:02d}_3d.npy".format(person_id, gesture_id),
            )
            skeleton_clip = np.load(filename_path, allow_pickle=True)

            for index, clip in enumerate(spots_data["labels"]):
                begin = clip["begin"]
                end = clip["end"]
                sliced = skeleton_clip[begin:end]
                spots.append((begin, end))

                # number_of_zeros = np.count_nonzero(sliced == 0)
                # if number_of_zeros > 0:
                #     continue

                # sliced_path = os.path.join(
                #     "./ifes_clips",
                #     "p{:03d}g{:02d}_r{:02d}_3d_new.npy".format(
                #         person_id, gesture_id, index
                #     ),
                # )
                # np.save(sliced_path, sliced, allow_pickle=True)

                # print(skeleton_clip.shape)
                # print(spots)

    return skeleton_clip, spots


# [T, M, V, C] = [tempo, pessoa, junta, coordenada] >> [begin:end, 0, :, :]
if __name__ == "__main__":
    split_clip()
