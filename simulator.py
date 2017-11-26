# Module simulator provides simulator of space environment
# and learning proccess of the agent.

# import numpy as np
import pykep as pk
import time
from api import Agent, Environment, SpaceObject


PARAMS = dict(coord=True, v=True)


def print_position(satellite, epoch):
    """ Print SpaceObject position. """
    pos, v = satellite.position(epoch)
    print("{} position: x - {:0.2f}, y - {:0.2f}, z - {:0.2f}.\
      \n{} velocity: Vx - {:0.2f}, Vy - {:0.2f}, Vz - {:0.2f}\
      \n".format(satellite.get_name(), pos[0], pos[1], pos[2],
                 satellite.get_name(), v[0], v[1], v[2]))


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

            print_position(self.environment.protected, self.curr_time)
            for obj in self.environment.debris:
                print_position(obj, self.curr_time)

            self.curr_time = pk.epoch(
                self.curr_time.mjd2000 + self.step, "mjd2000")
            time.sleep(3)


def main():
    pos = [1, 0, 1]
    # Time in format YYYY-MM-DD HH:MM:SS
    t = time.strftime("%Y-%m-%d %T")
    # Time in epoch format to use in pykep methods.
    epoch = pk.epoch_from_string(t)
    v = [1, 0, 1]
    mu = 1.0
    f = 1.0

    # ISS tle parameteres.
    iss1 = '1 25544U 98067A   17328.30743056  .00003472  00000-0  59488-4 0  9993'
    iss2 = '2 25544  51.6407 320.0980 0004356 149.0109  38.9204 15.54217487 86654'

    params_ISS = dict(tle_line1=iss1, tle_line2=iss2, f=f)
    params_debris = dict(pos=pos, v=v, epoch=epoch, mu=mu, f=f)

    agent = Agent()
    # Our protected object
    ISS = SpaceObject("ISS", True, params_ISS)
    # Space debris
    debris = [SpaceObject("D1", False, params_debris),
              SpaceObject("D2", False, params_debris)]
    env = Environment(ISS, debris)

    simulator = Simulator(agent, env)
    simulator.run()


if __name__ == "__main__":
    main()
