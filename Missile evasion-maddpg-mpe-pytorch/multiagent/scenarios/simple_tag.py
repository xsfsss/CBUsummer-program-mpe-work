"""
edit by Xu Ruicheng
2023.8
"""
import numpy as np
from multiagent.core import World, Agent, Landmark, Border
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 1
        num_adversaries = 2
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0
        num_borders = 80
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.05 if agent.adversary else 0.05
            agent.accel = 1.0 if agent.adversary else 1.5
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.2 if agent.adversary else 1.6
            """if '0' in agent.name:
                agent.layers = [1,1]
            elif '1' in agent.name:
                agent.layers = [1,2]
            elif '2' in agent.name:
                agent.layers = [2,2]"""
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.05
            landmark.boundary = False
        # make initial conditions
        # 加入 borders
        world.borders = [Border() for i in range(num_borders)]
        for i, border in enumerate(world.borders):
            border.name = 'border %d' % i
            border.collide = True
            border.movable = False
            border.size = 0.15  # 边界大小
            border.boundary = True
            # 改变边界厚度border.shape
            border.shape = [[-0.05, -0.05], [0.05, -0.05],
                            [0.05, 0.05], [-0.05, 0.05]]

        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, border in enumerate(world.borders):
            border.color = np.array([0.8, 0.4, 0.4])  # 边界颜色
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            #agent.state.p_pos = np.array([0.0, 0.0]) if agent.adversary else np.array([0.5, 0.5])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)#agent初始交流状态
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

        pos = []
        x = -0.95
        y = -1.0
        # bottom
        for count in range(20):
            pos.append([x, y])
            x += 0.1

        x = 1.0
        y = -0.95
        # right
        for count in range(20):
            pos.append([x, y])
            y += 0.1

        x = 0.95
        y = 1.0
        # top
        for count in range(20):
            pos.append([x, y])
            x -= 0.1

        x = -1.0
        y = 0.95
        # left
        for count in range(20):
            pos.append([x, y])
            y -= 0.1

        for i, border in enumerate(world.borders):
            border.state.p_pos = np.asarray(pos[i])  # 将设好的坐标传到border的位置坐标
            border.state.p_vel = np.zeros(world.dim_p)
            #if(i==79): print('done')

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10
        for i, border in enumerate(world.borders):
            if self.is_collision(border, agent):
                rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        """def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
            由于我设置了border所以这里的代码不需要
            """

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def observation(self, agent, world):
        #print('--------------------observation active--------------------')#确保被调用
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            #if not other.adversary:
            #    other_vel.append(other.state.p_vel)
            other_vel.append(other.state.p_vel)
        """if '2' not in agent.name:
            agent.size *= 1.01
            print(agent.name + ' ' + str(agent.size))"""

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    #这里多定义一个”导弹“的observation
    def adversary_observation(self, agent, world):
        #print('++++++++++++++adversary_observation active++++++++++++++')#确保被调用
        # ctl_pos = []
        # for ctl in self.controllers(world):
        # ctl_pos.append(ctl.state.p_pos - agent.state.p_pos)
        other_pos = []
        other_vel = []

        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)

        """if '2' not in agent.name:
            agent.size *= 1.01
            print(agent.name + ' ' + str(agent.size))"""
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + other_vel)


    def incr_size(self, world):
        for i, agent in enumerate(world.agents):
            agent.size *= 1.01 if agent.adversary else agent.size
        #return [agent.size for agent in world.agents if not agent.adversary]


