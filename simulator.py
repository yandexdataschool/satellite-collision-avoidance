# Module simulator provides simulator of space environment
# and learning proccess of the agent.
import numpy as np
import pykep as pk
import time
from api import Agent, Environment, SpaceObject


PARAMS = dict(coord=True, v=True)


class Simulator:
    """ Simulator allows to start the simulation of provided environment,
    and starts agent-environment collaboration.
    """

    def __init__(self, agent, environment, step=0.5):
        """
            agent -- Agent(), agent, to do actions in environment.
            environment -- Environment(), the initial space environment.
            step -- float, step in julian date.
        """
        self.is_end = False
        self.agent = agent
        self.environment = environment
        self.step = step
        self.start_time = pk.epoch_from_string(time.strftime("%Y-%m-%d %T"))
        self.curr_time = self.start_time

    def run(self):
        iteration = 0
        while not self.is_end:
            iteration += 1

            r = self.environment.get_reward()
            s = self.environment.get_state(PARAMS)
            action = self.agent.get_action(s, r)
            self.environment.act(action)

            self.is_end = self.environment.state.is_end

            print("Iter #{} \tEpoch: {}\tCollision: {}".format(
                iteration,  self.curr_time, self.is_end))

            self.curr_time = pk.epoch(
                self.curr_time.mjd2000 + self.step, "mjd2000")
            time.sleep(1)


def main():
    start_pos = np.zeros(3)

    # Time in format YYYY-MM-DD HH:MM:SS
    t = time.strftime("%Y-%m-%d %T")
    # Time in epoch format to use in pykep methods.
    start_t = pk.epoch_from_string(t)
    start_v = np.array([1, 0, 1])
    start_f = 1.0

    space_init = [start_pos, start_v, start_t, start_f]

    agent = Agent()
    # Our protected object
    ISS = SpaceObject(*space_init)
    # Space debris
    debris = [SpaceObject(*space_init),
              SpaceObject(*space_init)]
    env = Environment(ISS, debris)

    simulator = Simulator(agent, env)
    simulator.run()


if __name__ == "__main__":
    main()
