import argparse
import threading
import zmq

import carla.master.master_learner as master_learner
import carla.master.master_receiver as master_receiver


def main():
    parser = argparse.ArgumentParser(description='Reinforcement learning master')

    parser.add_argument('--frame_port', default=10666, help='Which port to wait for frames from workers')
    parser.add_argument('--network_port', default=10667, help='Which port to send networks to')
    parser.add_argument('--interface', default="tcp://*", help='Interface to bind to')
    parser.add_argument('--epochs', default=10, help='How many epochs to run')
    parser.add_argument('--envs', default=2, help='How many environments expect')
    parser.add_argument('--steps', default=5, help='How many steps from each environment')

    args = parser.parse_args()
    context = zmq.Context()
    print(args)

    learner_thread = threading.Thread(target=master_learner.main, args=(args, context))
    learner_thread.start()

    receiver_thread = threading.Thread(target=master_receiver.main, args=(args, context))
    receiver_thread.start()

    learner_thread.join()


if __name__ == '__main__':
    main()
