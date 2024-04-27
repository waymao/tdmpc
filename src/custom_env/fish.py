import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite import fish
from dm_control.utils import rewards
from dm_control.utils import io as resources
import numpy as np

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'custom_env')

_DEFAULT_TIME_LIMIT = 40
_CONTROL_TIMESTEP = .04
_JOINTS = ['tail1',
           'tail_twist',
           'tail2',
           'finright_roll',
           'finright_pitch',
           'finleft_roll',
           'finleft_pitch']

def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return resources.GetResource(os.path.join(_TASKS_DIR, 'fish.xml')), common.ASSETS

@fish.SUITE.add('custom')
def swim_dir(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Fish Swim task."""
  physics = fish.Physics.from_xml_string(*get_model_and_assets())
  task = Swim_dir(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit,
      **environment_kwargs)

@fish.SUITE.add('custom')
def swim_new(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Fish Swim task."""
  physics = fish.Physics.from_xml_string(*get_model_and_assets())
  task = Swim_new(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit,
      **environment_kwargs)

class Swim_new(fish.Swim):

  def __init__(self, random=None):

    super().__init__(random=random)
    target_angle = np.random.uniform(-np.pi, np.pi)
    self.swim_dir_x = np.cos(target_angle)
    self.swim_dir_y = np.sin(target_angle)

  def get_observation(self, physics):
    """Returns an observation of joints, target direction and velocities."""
    obs = collections.OrderedDict()
    obs['target'] = physics.mouth_to_target()
    obs['joint_angles'] = physics.joint_angles()
    obs['upright'] = physics.upright()
    obs['velocity'] = physics.velocity()
    obs['swim_dir', 'x'] = self.swim_dir_x
    obs['swim_dir', 'y'] = self.swim_dir_y
    obs['mouth_pos'] = physics.named.data.geom_xpos['mouth']
    obs['mouth_mat'] = physics.named.data.geom_xmat['mouth']
    return obs


class Swim_dir(fish.Swim):
  """A Fish `Task` for swimming with smooth reward."""

  def __init__(self, random=None):

    super().__init__(random=random)
    self._desired_speed = 2
    target_angle = np.random.uniform(-np.pi, np.pi)
    self.swim_dir_x = np.cos(target_angle)
    self.swim_dir_y = np.sin(target_angle)
    
  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""

    quat = self.random.randn(4)
    physics.named.data.qpos['root'][3:7] = quat / np.linalg.norm(quat)
    for joint in _JOINTS:
      physics.named.data.qpos[joint] = self.random.uniform(-.2, .2)
    # Randomize target position.
    physics.named.model.geom_pos['target', 'x'] = self.random.uniform(-.4, .4)
    physics.named.model.geom_pos['target', 'y'] = self.random.uniform(-.4, .4)
    physics.named.model.geom_pos['target', 'z'] = self.random.uniform(.1, .3)
    self.after_step(physics)

  def get_observation(self, physics):
    """Returns an observation of joints, target direction and velocities."""
    obs = collections.OrderedDict()
    obs['target'] = physics.mouth_to_target()
    obs['joint_angles'] = physics.joint_angles()
    obs['upright'] = physics.upright()
    obs['velocity'] = physics.velocity()
    obs['swim_dir', 'x'] = self.swim_dir_x
    obs['swim_dir', 'y'] = self.swim_dir_y
    obs['mouth_pos'] = physics.named.data.geom_xpos['mouth']
    obs['mouth_mat'] = physics.named.data.geom_xmat['mouth']
    return obs
  
  def get_reward(self, physics):
    """Returns a smooth reward."""
    speed = np.linalg.norm(physics.velocity()[:3])
    move_reward = rewards.tolerance(
                    speed,
                    bounds=(self._desired_speed, self._desired_speed*2),
                    margin=self._desired_speed)
    dir_ = np.array([self.swim_dir_x, self.swim_dir_y, 0])
    cos_dir = np.dot(physics.velocity()[:3]/speed, dir_)
    dir_reward = rewards.tolerance(
                    cos_dir,
                    bounds=(0.9, 1.0),
                    margin=1)
    is_upright = 0.5 * (physics.upright() + 1)
      
    return (dir_reward * move_reward * 7 + is_upright) / 8.0