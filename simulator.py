# Module simulator provides simulator of space environment
# and learning proccess of the agent.
import numpy as np
import time
from api import Agent, Environment, SpaceObject


PARAMS = dict(coord=True, v=True)


class Simulator:
    """ Simulator allows to start the simulation of provided environment,
    and starts agent-environment collaboration.
    """

    def __init__(self, agent, environment):
        """
            agent -- Agent(), agent, to do actions in environment.
            environment -- Environment(), the initial space environment.
        """
        self.is_end = False
        self.agent = agent
        self.environment = environment

    def run(self):
        step = 0
        start = time.time()

        while not self.is_end:
            step += 1

            r = self.environment.get_reward()
            s = self.environment.get_state(PARAMS)
            action = self.agent.get_action(s, r)
            self.environment.act(action)

            self.is_end = self.environment.state.is_end

            print 'Step #%d \tTime: %.2f \tNO Collision' % (step, time.time() - start)

            time.sleep(1)


if __name__ == "__main__":
    start_pos = np.zeros(3)
    start_v, start_t = 1, 0

    agent = Agent()
    # Our protected object
    ISS = SpaceObject(start_pos, start_v, start_t)
    # Space garbage
    garbage = [SpaceObject(start_pos, start_v, start_t),
               SpaceObject(start_pos, start_v, start_t)]
    env = Environment(ISS, garbage)

    simulator = Simulator(agent, env)
    simulator.run()
