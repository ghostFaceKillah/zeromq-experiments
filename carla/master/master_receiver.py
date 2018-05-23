import zmq
import collections

import carla.master.constants as cnst
import carla.messages.message_pb2 as pb

import collections


def main(args, context):
    # Internal socket for sending training batches
    batch_socket = context.socket(zmq.PAIR)
    batch_socket.connect(cnst.INTERNAL_INTERFACE)

    frame_socket = context.socket(zmq.REP)
    frame_socket.bind("{}:{}".format(args.interface, args.frame_port))

    clients_connected = 0

    print("Master receiver: initializing, awaiting clients")

    while clients_connected < args.envs:
        request = pb.InitializationRequest()
        request.ParseFromString(frame_socket.recv())
        print("Master receiver: client {} connected".format(clients_connected))
        frame_socket.send(pb.InitializationResponse(id=clients_connected, steps=args.steps).SerializeToString())
        clients_connected += 1

    print("All clients connected, ready to start")
    batch_socket.send_string('initialized')

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

        # TODO(jerry): This copy is quite painful to me and ideally I'd like to get rid of it
        batch_socket.send(batch.SerializeToString())

