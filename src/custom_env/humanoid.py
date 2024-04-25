import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base, common, humanoid
from dm_control.suite.humanoid import _WALK_SPEED, _CONTROL_TIMESTEP, _STAND_HEIGHT
from dm_control.utils import rewards
from dm_control.utils import io as resources
from dm_control.suite.utils import randomizers
import numpy as np

target_geom_xml = """
    <geom name="target" type="sphere" pos="0 5 1.4" size=".1" material="target"/>
    <light name="target_light" diffuse="1 1 1" pos="1 1 1.5"/>
""".encode()

@humanoid.SUITE.add('custom')
def walk_custom_obs(time_limit=25, random=None, environment_kwargs=None):
    """Returns the Stand task."""
    model, assets = humanoid.get_model_and_assets()
    # add target to the model
    model = model.replace(b"</worldbody>", target_geom_xml + b"</worldbody>")
    physics = Physics.from_xml_string(model, assets)
    task = HumanoidTargetObs(move_speed=_WALK_SPEED, pure_state=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)

@humanoid.SUITE.add('custom')
def walk_to_goal(time_limit=25, random=None, environment_kwargs=None):
    """Returns the Stand task."""
    model, assets = humanoid.get_model_and_assets()
    # add target to the model
    model = model.replace(b"</worldbody>", target_geom_xml + b"</worldbody>")
    physics = Physics.from_xml_string(model, assets)
    task = HumanoidTargetTask(move_speed=_WALK_SPEED, pure_state=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)

class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Walker domain."""

  def torso_upright(self):
    """Returns projection from z-axes of torso to the z-axes of world."""
    return self.named.data.xmat['torso', 'zz']

  def head_height(self):
    """Returns the height of the torso."""
    return self.named.data.xpos['head', 'z']

  def center_of_mass_position(self):
    """Returns position of the center-of-mass."""
    return self.named.data.subtree_com['torso'].copy()

  def center_of_mass_velocity(self):
    """Returns the velocity of the center-of-mass."""
    return self.named.data.sensordata['torso_subtreelinvel'].copy()

  def torso_vertical_orientation(self):
    """Returns the z-projection of the torso orientation matrix."""
    return self.named.data.xmat['torso', ['zx', 'zy', 'zz']]

  def joint_angles(self):
    """Returns the state without global orientation or position."""
    return self.data.qpos[7:].copy()  # Skip the 7 DoFs of the free root joint.
  
  def torso_to_target(self):
    """Returns a vector from nose to target in local coordinate of the head."""
    torso_to_target = (self.named.data.geom_xpos['target'] -
                        self.named.data.geom_xpos['torso'])
    torso_orientation = self.named.data.xmat['torso'].reshape(3, 3)
    return torso_to_target.dot(torso_orientation)[:2]
  
  def torso_to_target_dist(self):
    """Returns the distance from the torso to the target."""
    return np.linalg.norm(self.torso_to_target())

  def extremities(self):
    """Returns end effector positions in egocentric frame."""
    torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
    torso_pos = self.named.data.xpos['torso']
    positions = []
    for side in ('left_', 'right_'):
      for limb in ('hand', 'foot'):
        torso_to_limb = self.named.data.xpos[side + limb] - torso_pos
        positions.append(torso_to_limb.dot(torso_frame))
    return np.hstack(positions)


class HumanoidTargetObs(humanoid.Humanoid):
  def initialize_episode(self, physics: Physics):
    """Sets the state of the environment at the start of each episode."""
    randomizers.randomize_limited_and_rotational_joints(physics, self.random)

    # Randomize target position.
    close_target = self.random.rand() < .2  # Probability of a close target.
    target_box = .5 if close_target else 5
    xpos, ypos = self.random.uniform(-target_box, target_box, size=2)
    physics.named.model.geom_pos['target', 'x'] = xpos
    physics.named.model.geom_pos['target', 'y'] = ypos
    physics.named.model.light_pos['target_light', 'x'] = xpos
    physics.named.model.light_pos['target_light', 'y'] = ypos
    # physics.named.model.swim_dir['swim_dir', 'x'] = self.swim_dir_x
    # physics.named.model.swim_dir['swim_dir', 'y'] = self.swim_dir_y
    self.after_step(physics)


  def get_observation(self, physics: Physics):
    """Returns either the pure state or a set of egocentric features."""
    obs = collections.OrderedDict()
    if self._pure_state:
      obs['position'] = physics.position()
      obs['velocity'] = physics.velocity()
      obs['to_target'] = physics.torso_to_target()
    else:
      obs['joint_angles'] = physics.joint_angles()
      obs['head_height'] = physics.head_height()
      obs['extremities'] = physics.extremities()
      obs['torso_vertical'] = physics.torso_vertical_orientation()
      obs['com_velocity'] = physics.center_of_mass_velocity()
      obs['velocity'] = physics.velocity()
      obs['to_target'] = physics.torso_to_target()
    return obs


class HumanoidTargetTask(HumanoidTargetObs):
  """A humanoid task."""

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    standing = rewards.tolerance(physics.head_height(),
                                 bounds=(_STAND_HEIGHT, float('inf')),
                                 margin=_STAND_HEIGHT/4)
    upright = rewards.tolerance(physics.torso_upright(),
                                bounds=(0.9, float('inf')), sigmoid='linear',
                                margin=1.9, value_at_margin=0)
    stand_reward = standing * upright
    small_control = rewards.tolerance(physics.control(), margin=1,
                                      value_at_margin=0,
                                      sigmoid='quadratic').mean()
    small_control = (4 + small_control) / 5

    near_goal = rewards.tolerance(physics.torso_to_target_dist(),
                                    bounds=(0, 5), margin=5, value_at_margin=0,
                                    sigmoid='quadratic')
    if self._move_speed == 0:
      horizontal_velocity = physics.center_of_mass_velocity()[[0, 1]]
      dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
      return small_control * stand_reward * dont_move * near_goal
    else:
      com_velocity = np.linalg.norm(physics.center_of_mass_velocity()[[0, 1]])
      move = rewards.tolerance(com_velocity,
                               bounds=(self._move_speed, float('inf')),
                               margin=self._move_speed, value_at_margin=0,
                               sigmoid='linear')
      move = (5*move + 1) / 6
      return small_control * stand_reward * move * near_goal

if __name__ == '__main__':
    env = walk_to_goal()
    obs = env.reset()
    import numpy as np
    import time
    from matplotlib import pyplot as plt
    rew = []
    for i in range(15):
        next_obs, reward, done, info = env.step(np.zeros(21))
        img = env.physics.render()
        rew.append(reward)
        plt.imshow(img)
        plt.pause(0.01)
    print(np.array(rew).mean())
