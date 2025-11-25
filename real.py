import sys
import socket
from typing import Dict, Tuple

import cv2
import joblib
import colorsys
import warnings
import matplotlib.pyplot as plt
import matplotlib

# from norm import normalizacao
import pandas as pd
import numpy as np
import numpy.typing as npt
from is_msgs.image_pb2 import Image

# from is_project.detector import Detector
from google.protobuf.json_format import Parse

# from is_project.conf.options_pb2 import ServiceOptions
# is-wire lab visio >> pip install --user is-wire
from is_wire.core import Channel, Message, Subscription
from google.protobuf.message import Message as PbMessage
from is_msgs.image_pb2 import ObjectAnnotations
from is_msgs.image_pb2 import HumanKeypoints as HKP

from classificador import Gesture

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

TESTE = {
    0: "NOSE",
    1: "LEFT_EYE",
    2: "RIGHT_EYE",
    3: "LEFT_EAR",
    4: "RIGHT_EAR",
    5: "LEFT_SHOULDER",
    6: "RIGHT_SHOULDER",
    7: "LEFT_ELBOW",
    8: "RIGHT_ELBOW",
    9: "LEFT_WRIST",
    10: "RIGHT_WRIST",
    11: "LEFT_HIP",
    12: "RIGHT_HIP",
    13: "LEFT_KNEE",
    14: "RIGHT_KNEE",
    15: "LEFT_ANKLE",
    16: "RIGHT_ANKLE",
}

# (HKP.Value("RIGHT_KNEE")11
# HKP.Value("LEFT_EYE")17

links = [
    (HKP.Value("LEFT_SHOULDER"), HKP.Value("RIGHT_SHOULDER")),
    (HKP.Value("LEFT_SHOULDER"), HKP.Value("LEFT_HIP")),
    (HKP.Value("RIGHT_SHOULDER"), HKP.Value("RIGHT_HIP")),
    (HKP.Value("LEFT_HIP"), HKP.Value("RIGHT_HIP")),
    (HKP.Value("LEFT_SHOULDER"), HKP.Value("LEFT_EAR")),
    (HKP.Value("RIGHT_SHOULDER"), HKP.Value("RIGHT_EAR")),
    (HKP.Value("LEFT_SHOULDER"), HKP.Value("LEFT_ELBOW")),
    (HKP.Value("LEFT_ELBOW"), HKP.Value("LEFT_WRIST")),
    (HKP.Value("LEFT_HIP"), HKP.Value("LEFT_KNEE")),
    (HKP.Value("LEFT_KNEE"), HKP.Value("LEFT_ANKLE")),
    (HKP.Value("RIGHT_SHOULDER"), HKP.Value("RIGHT_ELBOW")),
    (HKP.Value("RIGHT_ELBOW"), HKP.Value("RIGHT_WRIST")),
    (HKP.Value("RIGHT_HIP"), HKP.Value("RIGHT_KNEE")),
    (HKP.Value("RIGHT_KNEE"), HKP.Value("RIGHT_ANKLE")),
    (HKP.Value("NOSE"), HKP.Value("LEFT_EYE")),
    (HKP.Value("LEFT_EYE"), HKP.Value("LEFT_EAR")),
    (HKP.Value("NOSE"), HKP.Value("RIGHT_EYE")),
    (HKP.Value("RIGHT_EYE"), HKP.Value("RIGHT_EAR")),
]

obj = Gesture()


def configure_plot(ax):
    """Configurações do gráfico 3D"""
    ax.clear()
    ax.view_init(azim=0, elev=20)
    ax.set_xlim(-4, 4)
    ax.set_xticks(np.arange(-4, 4, 1))
    ax.set_ylim(-4, 4)
    ax.set_yticks(np.arange(-4, 4, 1))
    ax.set_zlim(-0.25, 2)
    ax.set_zticks(np.arange(0, 2, 0.5))
    ax.set_xlabel("X", labelpad=20)
    ax.set_ylabel("Y", labelpad=10)
    ax.set_zlabel("Z", labelpad=5)


def _id_to_rgb_color(id) -> Tuple[float, float, float]:

    hue = (id % 20) / 20
    saturation = 0.8
    luminance = 0.6
    r, g, b = [x for x in colorsys.hls_to_rgb(hue, luminance, saturation)]

    return r, g, b


class StreamChannel(Channel):
    def __init__(
        self, uri: str = "amqp://guest:guest@localhost:5672", exchange: str = "is"
    ) -> None:
        super().__init__(uri=uri, exchange=exchange)

    """
    A class representing a streaming channel for consuming messages.

    Parameters:
    -----------
    uri : str, optional
        The URI for connecting to the message broker. 
        Defaults to "amqp://guest:guest@localhost:5672".

    exchange : str, optional
        The exchange to bind the channel to. Defaults to "is".

    Methods:
    --------
    consume_last() -> Tuple[Message, int]:
        Consume the last available message from the channel.

    Returns a tuple containing the consumed message and 
    the number of dropped messages.

    Examples:
    ---------
    >>> stream_channel = StreamChannel(uri="amqp://guest:guest@localhost:5672", exchange="is")
    >>> message, dropped_count = stream_channel.consume_last()
    """

    def consume_last(self) -> Tuple[Message, int]:
        """
        Consume the last available message from the channel.

        Returns:
        --------
        Tuple[Message, int]
            A tuple containing the consumed message and the number
            of dropped messages.
        """
        dropped = 0
        msg = super().consume()
        while True:
            try:
                # will raise an exception when no message remained
                msg = super().consume(timeout=0.0)
                dropped += 1
            except socket.timeout:
                return (msg, dropped)


def to_np(image: Image) -> npt.NDArray[np.uint8]:
    """
    Convert a Protocol Buffer Image message to a NumPy array.

    Parameters:
    -----------
    image : Image
        A Protocol Buffer Image message containing the encoded image.

    Returns:
    --------
    np.ndarray
        The NumPy array representing the decoded image.

    Examples:
    ---------
    >>> image_proto = SomeImageMessage
    >>> numpy_image = to_np(image_proto)
    """
    buffer = np.frombuffer(image.data, dtype=np.uint8)
    output = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    return output


def to_image(
    image: npt.NDArray[np.uint8],
    encode_format: str = ".jpeg",
    compression_level: float = 0.8,
) -> Image:
    """
    Convert a NumPy array representing an image to a Protocol Buffer Image message.

    Parameters:
    -----------
    image : np.ndarray
        The NumPy array representing the image.

    encode_format : str, optional
        The encoding format for the image. Defaults to ".jpeg".
        Supported formats: ".jpeg", ".png".

    compression_level : float, optional
        Compression level for the encoded image. Applicable for JPEG and PNG formats.
        For JPEG, the compression level ranges from 0.0 to 1.0
        (higher values mean higher quality).
        For PNG, the compression level ranges from 0.0 to 1.0
        (higher values mean higher compression).
        Defaults to 0.8.

    Returns:
    --------
    Image
        A Protocol Buffer Image message containing the encoded image.

    Examples:
    ---------
    >>> image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    >>> encoded_image = to_image(image_array, encode_format=".jpeg",
    compression_level=0.8)
    """
    if encode_format == ".jpeg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(compression_level * (100 - 0) + 0)]
    elif encode_format == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, int(compression_level * (9 - 0) + 0)]
    else:
        return Image()
    cimage = cv2.imencode(ext=encode_format, img=image, params=params)
    return Image(data=cimage[1].tobytes())


def load_json(filename: str, schema: PbMessage) -> PbMessage:
    """
    Load data from a JSON file and parse it into a Protocol Buffer message.

    Parameters:
    -----------
    filename : str
        The path to the JSON file to be loaded.

    schema : PbMessage
        An instance of the Protocol Buffer message schema that will be used for parsing.

    Returns:
    --------
    PbMessage
        A parsed instance of the Protocol Buffer message.

    Examples:
    ---------
    >>> schema = MyProtoMessage
    >>> loaded_message = load_json("data.json", schema)
    """
    with open(file=filename, mode="r", encoding="utf-8") as f:
        proto = Parse(f.read(), schema())
    return proto


def plt_reta(sk: np.ndarray, ax, y: int = 0, option: bool = False):

    rightx, righty, rightz, leftx, lefty, leftz = obj.reta_para_plot(sk)

    (
        y,
        left_distancia,
        left_teta,
        right_distancia,
        right_teta,
    ) = obj.classificador(sk, 0.25, 30, True)

    if y == 3 or y == 0:
        ax.plot(
            rightx,
            righty,
            rightz,
            color="g",
        )

        ax.plot(
            leftx,
            lefty,
            leftz,
            color="b",
        )

    elif y == 2:
        ax.plot(
            rightx,
            righty,
            rightz,
            color="g",
        )

    elif y == 1:
        ax.plot(
            leftx,
            lefty,
            leftz,
            color="b",
        )


def main():
    # Ignorar avisos de depreciação
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    """Plot - Matplot"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    service_name = "test"
    camera_id = 1
    address = "10.20.5.3:30000"
    canal = StreamChannel(f"amqp://guest:guest@{address}")
    assinatura = Subscription(canal, name=service_name)  ##
    for camera in [0, 1, 2, 3]:
        pass
    # assinatura.subscribe(topic=f"CameraGateway.{camera_id}.Frame")
    assinatura.subscribe(topic=f"SkeletonsGrouper.0.Localization")
    # assinatura.subscribe(topic=f"SkeletonsDetector.{camera_id}.Detection")
    i = 0

    arquivo_joblib = "logisticRegression.sav"
    randomForest = joblib.load(arquivo_joblib)

    while True:
        # ---------- Messagem Tempo Real ---------- #
        messagem, _ = canal.consume_last()
        results = messagem.unpack(ObjectAnnotations)

        # ---------- ---------------- ---------- #
        configure_plot(ax)

        # image = messagem.unpack(Image)
        # array = to_np(image)
        # print(results)

        # ---------- Obter posições Skeletons ---------- #
        # for skeleton in results.objects:
        #     skeletons: Dict = {}
        #     for part in skeleton.keypoints:
        #         skeletons[int(part.id)] = [
        #             part.position.x,
        #             part.position.y,
        #             part.position.z,
        #         ]

        # sk = [ ]

        # ---------- Escolher Skeleton mais próximo Origem ---------- #
        n_persons = len(results.objects)

        if n_persons == 1:
            pts = np.zeros((n_persons, 17, 3), dtype=np.float64)
            for i, skeleton in enumerate(results.objects):
                for part in skeleton.keypoints:
                    pts[i, TO_COCO_IDX[part.id], 0] = part.position.x
                    pts[i, TO_COCO_IDX[part.id], 1] = part.position.y
                    pts[i, TO_COCO_IDX[part.id], 2] = part.position.z

        elif n_persons > 1:
            pts = np.zeros((n_persons, 17, 3), dtype=np.float64)
            for i, skeleton in enumerate(results.objects):
                for part in skeleton.keypoints:
                    pts[i, TO_COCO_IDX[part.id], 0] = part.position.x
                    pts[i, TO_COCO_IDX[part.id], 1] = part.position.y
                    pts[i, TO_COCO_IDX[part.id], 2] = part.position.z
            # mean of all joints
            origin = np.zeros((n_persons, 3))
            pts_mean = np.mean(pts, axis=1)
            distance = np.linalg.norm(pts_mean - origin, axis=1)
            # closer to origin
            idx = distance.argmin()
            pts = pts[idx, :, :]
            pts = pts.reshape(1, 17, 3)

        else:
            continue  # pts = np.zeros((1, 17, 3), dtype=np.float64)

        # ---------- Normalização Skeletons ---------- #
        skM = obj.normalizacao(pts[0])

        # ---------- Predict RandomForest ---------- #
        df = obj.list_to_dataframe(skM)
        predict = randomForest.predict(df.loc[[0]])
        predict = predict[0]

        # ---------- Desenho Reta Matplot ---------- #
        if predict == 1:
            plt_reta(pts[0], ax)

        # ---------- Desenho Skeletons Matplot ---------- #
        for link in links:
            begin, end = link

            x_pair = [
                pts[0][TO_COCO_IDX[begin]][0],
                pts[0][TO_COCO_IDX[end]][0],
            ]
            y_pair = [
                pts[0][TO_COCO_IDX[begin]][1],
                pts[0][TO_COCO_IDX[end]][1],
            ]
            z_pair = [
                pts[0][TO_COCO_IDX[begin]][2],
                pts[0][TO_COCO_IDX[end]][2],
            ]
            color = "r"  # _id_to_rgb_color(id=46)
            ax.plot(
                x_pair,
                y_pair,
                zs=z_pair,
                linewidth=3,
                color=color,
            )
            # ax.text(
            #     x=-4,
            #     y=-4,
            #     z=3,
            #     s=f"{predict}",
            #     fontsize=12,
            # )"

        # ---------- Plot view cv2 ---------- #
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape((height, width, 4))
        img_rgb = img[:, :, 1:4]

        # ---------- Encerrar Tecla "q" ---------- #
        cv2.imshow("Skeletons", img_rgb)
        key = cv2.waitKey(1)
        if key == ord("q"):
            return


if __name__ == "__main__":
    main()
