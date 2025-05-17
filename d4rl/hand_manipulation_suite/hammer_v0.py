import numpy as np
from gym import utils
from gym import spaces
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
from d4rl.utils.quatmath import quat2euler
from d4rl import offline_env
import os

ADD_BONUS_REWARDS = True

class HammerEnvV0(mujoco_env.MujocoEnv, utils.EzPickle, offline_env.OfflineEnv):
    def __init__(self, reward_type='dense_l2', hammer_reward_type='dense_l2', nail_reward_type='dense_l2', velocity_reward_type='dense_l2', **kwargs):
        offline_env.OfflineEnv.__init__(self, **kwargs)
        self.reward_type = reward_type
        self.hammer_reward_type = hammer_reward_type
        self.nail_reward_type = nail_reward_type
        self.velocity_reward_type = velocity_reward_type
        self.target_obj_sid = -1
        self.S_grasp_sid = -1
        self.obj_bid = -1
        self.tool_sid = -1
        self.goal_sid = -1
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_hammer.xml', 5)

        # Override action_space to -1, 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, dtype=np.float32, shape=self.action_space.shape)

        utils.EzPickle.__init__(self)

        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])
        
        self.target_obj_sid = self.sim.model.site_name2id('S_target')
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        self.tool_sid = self.sim.model.site_name2id('tool')
        self.goal_sid = self.sim.model.site_name2id('nail_goal')
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a * self.act_rng  # mean center and scale
        except:
            a = a  # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        tool_pos = self.data.site_xpos[self.tool_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        goal_pos = self.data.site_xpos[self.goal_sid].ravel()
        
        # compute distance between tool and target
        d = np.linalg.norm(tool_pos - target_pos)
        d_og_l1 = np.sum(np.abs(tool_pos - target_pos))
        
        # compute reward based on reward_type
        if self.reward_type == "sparse":
            # sparse reward: returns -1 if distance is above threshold, 0 otherwise
            reward = -np.array(d > 0.1, dtype=np.float32)
        elif self.reward_type == "dense_l2":
            # dense L2 reward: negative Euclidean distance
            reward = -d.astype(np.float32)
        elif self.reward_type == 'dense_l1':
            # dense L1 reward: negative Manhattan distance
            reward = -d_og_l1.astype(np.float32)
        elif self.reward_type == 'dense_l2_exp':
            # dense L2 exponential reward: exp((1-d)*10)
            reward = np.exp((1-d)*10).astype(np.float32)
        elif self.reward_type == 'dense_l2_log':
            # dense L2 logarithmic reward: log((10-d)*10)
            reward = np.log((10-d)*10).astype(np.float32)
        elif self.reward_type == 'dense_l2_plateau':
            # dense L2 plateau reward: -exp(-(d-10))
            reward = -np.exp(-(d-10)).astype(np.float32)
        else:
            # default dense L2 reward
            reward = -d.astype(np.float32)
        
        # compute hammer reward
        hammer_dist = np.linalg.norm(palm_pos - obj_pos)
        if self.hammer_reward_type == "sparse":
            hammer_reward = -np.array(hammer_dist > 0.1, dtype=np.float32)
        elif self.hammer_reward_type == "dense_l2":
            hammer_reward = -0.1 * hammer_dist.astype(np.float32)
        elif self.hammer_reward_type == 'dense_l1':
            hammer_reward = -0.1 * np.sum(np.abs(palm_pos - obj_pos)).astype(np.float32)
        elif self.hammer_reward_type == 'dense_l2_exp':
            hammer_reward = 0.1 * np.exp((1-hammer_dist)*10).astype(np.float32)
        elif self.hammer_reward_type == 'dense_l2_log':
            hammer_reward = 0.1 * np.log((10-hammer_dist)*10).astype(np.float32)
        elif self.hammer_reward_type == 'dense_l2_plateau':
            hammer_reward = -0.1 * np.exp(-(hammer_dist-10)).astype(np.float32)
        else:
            hammer_reward = -0.1 * hammer_dist.astype(np.float32)
        
        # compute nail reward
        nail_dist = np.linalg.norm(target_pos - goal_pos)
        if self.nail_reward_type == "sparse":
            nail_reward = -np.array(nail_dist > 0.1, dtype=np.float32)
        elif self.nail_reward_type == "dense_l2":
            nail_reward = -10 * nail_dist.astype(np.float32)
        elif self.nail_reward_type == 'dense_l1':
            nail_reward = -10 * np.sum(np.abs(target_pos - goal_pos)).astype(np.float32)
        elif self.nail_reward_type == 'dense_l2_exp':
            nail_reward = 10 * np.exp((1-nail_dist)*10).astype(np.float32)
        elif self.nail_reward_type == 'dense_l2_log':
            nail_reward = 10 * np.log((10-nail_dist)*10).astype(np.float32)
        elif self.nail_reward_type == 'dense_l2_plateau':
            nail_reward = -10 * np.exp(-(nail_dist-10)).astype(np.float32)
        else:
            nail_reward = -10 * nail_dist.astype(np.float32)
        
        # compute velocity penalty
        velocity = np.linalg.norm(self.data.qvel.ravel())
        if self.velocity_reward_type == "sparse":
            velocity_reward = -np.array(velocity > 0.1, dtype=np.float32)
        elif self.velocity_reward_type == "dense_l2":
            velocity_reward = -1e-2 * velocity.astype(np.float32)
        elif self.velocity_reward_type == 'dense_l1':
            velocity_reward = -1e-2 * np.sum(np.abs(self.data.qvel.ravel())).astype(np.float32)
        elif self.velocity_reward_type == 'dense_l2_exp':
            velocity_reward = 1e-2 * np.exp((1-velocity)*10).astype(np.float32)
        elif self.velocity_reward_type == 'dense_l2_log':
            velocity_reward = 1e-2 * np.log((10-velocity)*10).astype(np.float32)
        elif self.velocity_reward_type == 'dense_l2_plateau':
            velocity_reward = -1e-2 * np.exp(-(velocity-10)).astype(np.float32)
        else:
            velocity_reward = -1e-2 * velocity.astype(np.float32)
        
        # combine rewards
        reward += hammer_reward + nail_reward + velocity_reward

        if ADD_BONUS_REWARDS:
            # bonus for lifting up the hammer
            if obj_pos[2] > 0.04 and tool_pos[2] > 0.04:
                reward += 2

            # bonus for hammering the nail
            if (np.linalg.norm(target_pos - goal_pos) < 0.020):
                reward += 25
            if (np.linalg.norm(target_pos - goal_pos) < 0.010):
                reward += 75

        goal_achieved = True if np.linalg.norm(target_pos - goal_pos) < 0.010 else False

        return ob, reward, False, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        qv = np.clip(self.data.qvel.ravel(), -1.0, 1.0)
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        obj_rot = quat2euler(self.data.body_xquat[self.obj_bid].ravel()).ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        nail_impact = np.clip(self.sim.data.sensordata[self.sim.model.sensor_name2id('S_nail')], -1.0, 1.0)
        return np.concatenate([qp[:-6], qv[-6:], palm_pos, obj_pos, obj_rot, target_pos, np.array([nail_impact])])

    def reset_model(self):
        self.sim.reset()
        target_bid = self.model.body_name2id('nail_board')
        self.model.body_pos[target_bid,2] = self.np_random.uniform(low=0.1, high=0.25)
        self.sim.forward()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        board_pos = self.model.body_pos[self.model.body_name2id('nail_board')].copy()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel().copy()
        return dict(qpos=qpos, qvel=qvel, board_pos=board_pos, target_pos=target_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        board_pos = state_dict['board_pos']
        self.set_state(qp, qv)
        self.model.body_pos[self.model.body_name2id('nail_board')] = board_pos
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 45
        self.viewer.cam.distance = 2.0
        self.sim.forward()

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if nail insude board for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
