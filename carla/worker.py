import argparse
import zmq
import gym

import carla.messages.message_pb2 as pb

class WorkerConnection:
    def __init__(self, frame_port, network_port, interface):
        context = zmq.Context()

        self.network_socket = context.socket(zmq.SUB)
        self.network_socket.connect('{}:{}'.format(interface, network_port))
        self.network_socket.setsockopt(zmq.SUBSCRIBE, b'')

        self.frame_socket = context.socket(zmq.REQ)
        self.frame_socket.connect('{}:{}'.format(interface, frame_port))

        # received during initialization dance
        self.config = None
        self.client_id = None

    def initialization_dance(self):
        initialization_request = pb.InitializationRequest()
        self.frame_socket.send(initialization_request.SerializeToString())

        print("Worker: awaiting initialization")
        response = pb.InitializationResponse()
        response.ParseFromString(self.frame_socket.recv())
        print("Worker: initialized")

        self.config = response
        self.client_id = self.config.id

        return response

    def receive_network(self):
        print("Worker: awaiting network")
        network = pb.Network()
        network.ParseFromString(self.network_socket.recv())
        print("Worker: received network version: {}".format(network.version))

        return network

    def send_frame(self, observation, reward, action, value, network_version):
        frame = pb.Frame(
            observation=observation.tobytes(),
            reward=reward,
            action=action,
            value=value,
            client_id=self.client_id
        )

        print("Worker: sending frame with network version: {}".format(network_version))
        self.frame_socket.send(frame.SerializeToString())
        self.frame_socket.recv()  # Discard the response


def policy(env):
    return env.action_space.sample(), 0.0


def main():
    """ Initialize and run worker program. Connect to master and download initialization parameters """
    parser = argparse.ArgumentParser(description='Reinforcement learning worker')

    parser.add_argument('--frame_port', default=10666, help='Which port to wait for frames from workers')
    parser.add_argument('--network_port', default=10667, help='Which port to send networks to')
    parser.add_argument('--interface', default="tcp://localhost", help='Interface to connect to')

    args = parser.parse_args()

    connection = WorkerConnection(
        frame_port=args.frame_port,
        network_port=args.network_port,
        interface=args.interface
    )

    config = connection.initialization_dance()
    env_steps = config.steps

    env = gym.make("BreakoutNoFrameskip-v4")
    env.reset()

    while True:
        network = connection.receive_network()

        # This is how many steps of the env the server wants
        for i in range(env_steps):
            next_action, current_value = policy(env)
            observation, reward, done, info = env.step(next_action)

            connection.send_frame(
                observation=observation,
                reward=reward,
                action=next_action,
                value=current_value,
                network_version=network.version
            )


if __name__ == '__main__':
    main()
