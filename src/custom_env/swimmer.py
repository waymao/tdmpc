import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite import swimmer
from dm_control.utils import rewards
from dm_control.utils import io as resources
from dm_control.suite.utils import randomizers
import numpy as np

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tasks')

_DEFAULT_TIME_LIMIT = 30
_CONTROL_TIMESTEP = .03  # (Seconds)


@swimmer.SUITE.add('custom')
def swim6_dir(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns a 6-link swimmer."""
  return _make_swimmer(6, time_limit, random=random,
                       environment_kwargs=environment_kwargs)

def _make_swimmer(n_joints, time_limit=_DEFAULT_TIME_LIMIT, random=None,
                  environment_kwargs=None):
  """Returns a swimmer control environment."""
  model_string, assets = swimmer.get_model_and_assets(n_joints)
  physics = Physics.from_xml_string(model_string, assets=assets)
  task = Swimmer_swim6_dir(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

class Physics(mujoco.Physics):
  """Physics simulation with additional features for the swimmer domain."""

  def nose_to_target(self):
    """Returns a vector from nose to target in local coordinate of the head."""
    nose_to_target = (self.named.data.geom_xpos['target'] -
                      self.named.data.geom_xpos['nose'])
    head_orientation = self.named.data.xmat['head'].reshape(3, 3)
    return nose_to_target.dot(head_orientation)[:2]

  def nose_to_target_dist(self):
    """Returns the distance from the nose to the target."""
    return np.linalg.norm(self.nose_to_target())

  def body_velocities(self):
    """Returns local body velocities: x,y linear, z rotational."""
    xvel_local = self.data.sensordata[12:].reshape((-1, 6))
    vx_vy_wz = [0, 1, 5]  # Indices for linear x,y vels and rotational z vel.
    return xvel_local[:, vx_vy_wz].ravel()

  def joints(self):
    """Returns all internal joint angles (excluding root joints)."""
    return self.data.qpos[3:].copy()


class Swimmer_swim6_dir(swimmer.Swimmer):
  """A Fish `Task` for swimming with smooth reward."""

  def __init__(self, random=None):

    super().__init__(random=random)

    target_angle = np.random.uniform(-np.pi, np.pi)
    self.swim_dir_x = np.cos(target_angle)
    self.swim_dir_y = np.sin(target_angle)
    self._desired_speed = 5

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    randomizers.randomize_limited_and_rotational_joints(physics, self.random)


    # Randomize target position.
    close_target = self.random.rand() < .2  # Probability of a close target.
    target_box = .3 if close_target else 2
    xpos, ypos = self.random.uniform(-target_box, target_box, size=2)
    physics.named.model.geom_pos['target', 'x'] = xpos
    physics.named.model.geom_pos['target', 'y'] = ypos
    physics.named.model.light_pos['target_light', 'x'] = xpos
    physics.named.model.light_pos['target_light', 'y'] = ypos
    physics.named.model.swim_dir['swim_dir', 'x'] = self.swim_dir_x
    physics.named.model.swim_dir['swim_dir', 'y'] = self.swim_dir_y
    self.after_step(physics)

  def get_observation(self, physics):
    """Returns an observation of joints, target direction and velocities."""
    obs = collections.OrderedDict()
    obs['joints'] = physics.joints()
    obs['body_velocities'] = physics.body_velocities()
    obs['to_target']  = physics.nose_to_target()
    obs['swim_dir'] = physics.named.model.swim_dir['swim_dir']

    return obs

  def get_reward(self, physics):
      """Returns a smooth reward."""
      move_reward = rewards.tolerance(
                    physics.body_velocities()[0],
                    bounds=(self._desired_speed, float('inf')),
                    margin=self._desired_speed,
                    value_at_margin=0.5,
                    sigmoid='linear')
      dir_reward = 1000.0 * np.dot(np.linalg.norm(self.body_velocities()), 
                                 physics.named.model.swim_dir['swim_dir'])
      return (move_reward + dir_reward) / 2.0


if __name__ == '__main__':
    env = swim6_dir(multitask=True)
    obs = env.reset()
    import numpy as np
    next_obs, reward, done, info = env.step(np.zeros(5))
    print(reward)