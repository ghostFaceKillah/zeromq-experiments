import argparse
import zmq
import gym

import carla.messages.message_pb2 as pb


def main():
    """ Initialize and run worker program. Connect to master and download initialization parameters """
    parser = argparse.ArgumentParser(description='Reinforcement learning worker')

    parser.add_argument('--frame_port', default=10666, help='Which port to wait for frames from workers')
    parser.add_argument('--network_port', default=10667, help='Which port to send networks to')
    parser.add_argument('--interface', default="tcp://localhost", help='Interface to connect to')

    args = parser.parse_args()
    context = zmq.Context()

    network_socket = context.socket(zmq.SUB)
    network_socket.connect('{}:{}'.format(args.interface, args.network_port))
    network_socket.setsockopt(zmq.SUBSCRIBE, b'')

    frame_socket = context.socket(zmq.REQ)
    frame_socket.connect('{}:{}'.format(args.interface, args.frame_port))

    initialization_request = pb.InitializationRequest()

    frame_socket.send(initialization_request.SerializeToString())

    print("Worker: awaiting initialization")
    response = pb.InitializationResponse()
    response.ParseFromString(frame_socket.recv())
    print("Worker: initialized")

    worker_function(response, network_socket, frame_socket)


def policy(env):
    return env.action_space.sample(), 0.0


def worker_function(initialization, network_socket, frame_socket):
    """
    Download a network and perform learning updates
    """
    env = gym.make("BreakoutNoFrameskip-v4")
    env.reset()

    while True:
        print("Worker: awaiting network")
        network = pb.Network()
        network.ParseFromString(network_socket.recv())
        print("Worker: received network version: {}".format(network.version))

        # This is how many steps of the env the server wants
        for i in range(initialization.steps):
            next_action, current_value = policy(env)
            observation, reward, done, info = env.step(next_action)

            frame = pb.Frame(
                observation=observation.tobytes(),
                reward=reward,
                action=next_action,
                value=current_value,
                client_id=initialization.id
            )

            print("Worker: sending frame with network version: {}".format(network.version))
            frame_socket.send(frame.SerializeToString())
            frame_socket.recv()  # Discard the response


if __name__ == '__main__':
    main()
