import argparse
import collections
import zmq

import carla.messages.message_pb2 as pb


def main():
    parser = argparse.ArgumentParser(description='Reinforcement learning master')

    parser.add_argument('--frame_port', default=10666, help='Which port to wait for frames from workers')
    parser.add_argument('--network_port', default=10667, help='Which port to send networks to')
    parser.add_argument('--interface', default="tcp://*", help='Interface to bind to')
    parser.add_argument('--epochs', default=10, help='How many epochs to run')
    parser.add_argument('--envs', default=1, help='How many environments expect')
    parser.add_argument('--steps', default=5, help='How many steps from each environment')

    args = parser.parse_args()
    context = zmq.Context()
    print(args)

    frame_socket = context.socket(zmq.REP)
    frame_socket.bind("{}:{}".format(args.interface, args.frame_port))

    network_socket = context.socket(zmq.PUB)
    network_socket.bind("{}:{}".format(args.interface, args.network_port))

    clients_connected = 0

    print("Master receiver: initializing, awaiting clients")

    while clients_connected < args.envs:
        request = pb.InitializationRequest()
        request.ParseFromString(frame_socket.recv())
        print("Master receiver: client {} connected".format(clients_connected))
        frame_socket.send(pb.InitializationResponse(id=clients_connected, steps=args.steps).SerializeToString())
        clients_connected += 1

    print("All clients connected, ready to start")

    print("Master learner: initialized, sending initial network")
    network_version = 0

    network = pb.Network(version=network_version)
    network_socket.send(network.SerializeToString())

    # Wait for the frames from the clients
    while True:
        frames = collections.defaultdict(lambda: collections.deque(maxlen=args.steps))
        client_counters = collections.defaultdict(int)
        is_batch_ready = False

        while not is_batch_ready:
            frame = pb.Frame()
            frame.ParseFromString(frame_socket.recv())

            frames[frame.client_id].append(frame)
            client_counters[frame.client_id] = len(frames[frame.client_id])

            print("Received frame from client: {} [count={}]".format(frame.client_id, client_counters[frame.client_id]))

            response = pb.FrameResponse()
            frame_socket.send(response.SerializeToString())

            if sum(client_counters.values()) == args.steps * args.envs:
                is_batch_ready = True

        batch = pb.FrameBatch()

        for i in range(args.envs):
            batch.frame.extend(frames[i])

        print("Final buffer ready, length: {}".format(len(batch.frame)))

        print("Learner received batch of size: {}".format(len(batch.frame)))
        print("Learning...")
        print("Sending network update")
        network_version += 1
        network = pb.Network(version=network_version)
        network_socket.send(network.SerializeToString())







if __name__ == '__main__':
    main()
