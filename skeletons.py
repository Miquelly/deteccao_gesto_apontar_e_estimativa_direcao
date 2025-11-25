import socket
from typing import Tuple
import matplotlib.pyplot as plt
from is_msgs.image_pb2 import ObjectAnnotations
from is_msgs.image_pb2 import HumanKeypoints as HKP
from is_wire.core import Channel, Subscription, Message


class StreamChannel(Channel):
    def __init__(
        self, uri: str = "amqp://guest:guest@localhost:5672", exchange: str = "is"
    ) -> None:
        super().__init__(uri=uri, exchange=exchange)

    def consume_last(self) -> Tuple[Message, int]:
        dropped = 0
        msg = super().consume()
        while True:
            try:
                # will raise an exception when no message remained
                msg = super().consume(timeout=0.0)
                dropped += 1
            except socket.timeout:
                return (msg, dropped)


def main():

    camera_id = 1
    service_name = "test"
    channel = StreamChannel(f"amqp://10.20.5.3:30000")
    # channel = Channel(f"amqp://10.20.5.2:30000")  ## "broker_uri": "amqp://10.20.5.2:30000"
    assinatura = Subscription(channel, name=service_name)
    assinatura.subscribe(topic=f"SkeletonsGrouper.0.Localization")

    while True:

        messagem, _ = channel.consume_last()
        results = messagem.unpack(ObjectAnnotations)
        print(results)


if __name__ == "__main__":
    main()
