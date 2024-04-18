import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base, common, cheetah
from dm_control.utils import rewards
from dm_control.utils import io as resources
import numpy as np

@cheetah.SUITE.add('custom')
def target_obs(time_limit=10, random=None, environment_kwargs=None):
    """Returns the Cheetah target task."""
    physics = Physics.from_xml_string(*cheetah.get_model_and_assets())
    task = CheetahTarget(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Cheetah domain."""
    def __init__(self, target_pos: list = [0, 0, 0.7]):
        self.target_pos = target_pos

    def torso_to_target(self):
        """Returns a vector from nose to target in local coordinate of the head."""
        nose_to_target = (self.target_pos -
                        self.named.data.geom_xpos['torso'])
        head_orientation = self.named.data.xmat['head'].reshape(3, 3)
        return nose_to_target.dot(head_orientation)[:2]

    def torso_to_target_dist(self):
        """Returns the distance from the nose to the target."""
        return np.linalg.norm(self.nose_to_target())

    def speed(self):
        """Returns the horizontal speed of the Cheetah."""
        return self.named.data.sensordata['torso_subtreelinvel'][0]


class CheetahTarget(cheetah.Cheetah):
    def get_observation(self, physics):
        obs = super().get_observation(physics)
        obs['to_target'] = physics.nose_to_target()
        return obs
    
    def get_reward(self, physics):
        return super().get_reward(physics)
