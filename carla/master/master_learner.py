import zmq

import carla.master.constants as cnst
import carla.messages.message_pb2 as pb


def main(args, context):
    # Internal socket for receiving training batches
    batch_socket = context.socket(zmq.PAIR)
    batch_socket.bind(cnst.INTERNAL_INTERFACE)

    network_socket = context.socket(zmq.PUB)
    network_socket.bind("{}:{}".format(args.interface, args.network_port))

    print("Master learner: awaiting initialization")
    assert batch_socket.recv_string() == 'initialized'
    print("Master learner: initialized, sending initial network")
    network_version = 0

    network = pb.Network(version=network_version)
    network_socket.send(network.SerializeToString())

    for i in range(1, args.epochs+1):
        print("Master learner: awaiting data for epoch {}".format(i))
        batch = pb.FrameBatch()
        batch.ParseFromString(batch_socket.recv())

        print("Learner received batch of size: {}".format(len(batch.frame)))
        print("Learning...")
        print("Sending network update")
        network_version += 1
        network = pb.Network(version=network_version)
        network_socket.send(network.SerializeToString())

    print("==============================")
    print("Learning is finished")
    print("==============================")

    batch_socket.close()
    network_socket.close()

