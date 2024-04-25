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
    _, assets = cheetah.get_model_and_assets()
    model = common.read_model(os.path.dirname(__file__) + os.sep + 'cheetah.xml')
    physics = Physics.from_xml_string(model, assets)
    task = CheetahTarget(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Cheetah domain."""

    def torso_to_target(self):
        """Returns a vector from nose to target in local coordinate of the head."""
        torso_to_target = (self.named.data.geom_xpos['target'] -
                        self.named.data.geom_xpos['torso'])
        torso_orientation = self.named.data.xmat['torso'].reshape(3, 3)
        return torso_to_target.dot(torso_orientation)[:2]

    def torso_to_target_dist(self):
        """Returns the distance from the nose to the target."""
        return np.linalg.norm(self.nose_to_target())

    def speed(self):
        """Returns the horizontal speed of the Cheetah."""
        return self.named.data.sensordata['torso_subtreelinvel'][0]


class CheetahTarget(cheetah.Cheetah):
    def get_observation(self, physics):
        obs = super().get_observation(physics)
        obs['to_target'] = physics.torso_to_target()
        return obs
    
    def get_reward(self, physics):
        return super().get_reward(physics)
