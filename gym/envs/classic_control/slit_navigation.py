import numpy as np
import gym
from pdb import set_trace as bp
import sys


class Slit_Navigation(gym.Env):
    def __init__(self,
                 init_cond=[0.0, 0.5],
                 slit_width=0.1,
                 slit_loc=[3.0, 0.0],
                 vx=1.0,
                 action_scale=2.0,
                 deltat=0.01,
                 target_loc=[4.0, 0.5],
                 target_rad=0.1):

        self.slit_width = slit_width
        self.slit_loc = slit_loc
        self.eps = 0.01  # wall width
        self.init_cond = np.array(init_cond)  # initial condition
        self.vx = vx
        self.action_scale = action_scale
        self.dt = deltat
        self.target_loc = target_loc
        self.target_rad = target_rad
        self.N_steps = ((self.target_loc[0] - self.init_cond[0]) / self.vx) / self.dt
        # print('N_steps should be ', self.N_steps)

        action_dim = 1
        self.reset()
        obs_dim = len(self.state)
        obs_dummy = np.array([1.] * obs_dim)
        self.action_space = gym.spaces.Box(low=np.array([-1.] * action_dim), high=np.array([1.] * action_dim))
        self.observation_space = gym.spaces.Box(low=-np.inf * obs_dummy, high=np.inf * obs_dummy)

    def reset(self):
        self.cur_step = 0
        self.state = self.init_cond.copy()
        return np.array(self.state)

    def check_collision(self, x, y):
        col = False
        if abs(x - self.slit_loc[0]) < self.eps:  # x_wall -eps < x < x_wall+eps
            if (y > self.slit_loc[1] + self.slit_width / 2.0) or (y < self.slit_loc[1] - self.slit_width / 2.0):
                col = True
        return col

    def get_reward(self, x, y):
        r = 0.0
        d = np.array([x, y]) - np.array(self.target_loc)
        if np.linalg.norm(d) < self.target_rad:
            r = 10.0
        return r

    def step(self, action):
        if action is None:
            self.act = 0.0
        else:
            if isinstance(action, np.ndarray):
                self.act = action[0] * self.action_scale
            else:
                self.act = action * self.action_scale

        x = self.state[0]
        y = self.state[1]

        reward = 0.0
        col = self.check_collision(x, y)
        if col:
            reward = -1.0
        # stuck there! no dynamics
        else:
            x = x + self.vx * self.dt
            y = y + self.act * self.dt
            reward = self.get_reward(x, y)

        self.state = [x, y]
        # print('state',self.state)
        # print('reward',reward)
        # reward += 0.1

        self.cur_step += 1

        # if self.cur_step >= 400 and self.state[0] < 3.5:
        #     # print(self.cur_step)
        #     reward -= 10.0

        return np.array(self.state), reward, False, {}

    def render(self, traj_data):

        X = np.array(traj_data[0])
        returns = np.array(traj_data[1])

        # This plots a trajectory X (should be a 2D-array)
        from matplotlib import pyplot as plt
        Barrier1 = np.array([-2.0, self.slit_loc[1] - self.slit_width / 2.0])
        Barrier2 = np.array([self.slit_loc[1] + self.slit_width / 2.0, 2.0])

        # t_obst = ((self.slit_loc[0]-self.init_cond[0])/self.vx)/self.dt
        # t_barr = np.array([t_obst,t_obst])
        # disk = plt.Circle((self.N_steps,self.target_loc[1]),radius =self.target_rad, color = 'blue',clip_on=False )
        # plt.close('all')
        # plt.figure(1)
        # plt.plot(X[:,1],'-r',linewidth = 3)
        # plt.plot(t_barr,Barrier1,'-k',linewidth = 3)
        # plt.plot(t_barr,Barrier2,'-k',linewidth = 3)
        # fig = plt.gcf()
        # ax = fig.gca()
        # ax.add_artist(disk)
        # plt.show()

        disk = plt.Circle((self.target_loc[0], self.target_loc[1]), radius=self.target_rad, color='blue', clip_on=False)
        plt.close('all')
        plt.figure(1)
        if np.ndim(X) == 2:
            plt.plot(X[:, 0], X[:, 1], '-r', linewidth=2)
        elif np.ndim(X) == 3:
            colors = (np.array(returns) - np.min(returns)) / (np.max(returns) - np.min(returns) + 0.0001)
            colors = np.concatenate([[colors], [1-colors], [np.ones(len(colors))], [np.ones(len(colors))]], axis=0).T

            for i, data in enumerate(X):
                plt.plot(data[:, 0], data[:, 1], '-', linewidth=2, alpha=0.3, c=colors[i])
        plt.plot(np.array([self.slit_loc[0], self.slit_loc[0]]), Barrier1, '-k', linewidth=3)
        plt.plot(np.array([self.slit_loc[0], self.slit_loc[0]]), Barrier2, '-k', linewidth=3)
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(disk)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([-0.25, 4.25])
        plt.title('Slit width: ' + str(self.slit_width))
        plt.show()
    # plt.savefig('traj.png')











