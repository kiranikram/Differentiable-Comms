class JungleDemoEnv(gym.Env, EzPickle):
    def __init__(self, env_config):
        EzPickle.__init__(self)
        self.seed(1)

        self.size = env_config.get('grid_size')

        self.observation_space = spaces.Dict(
            {
                "agents": spaces.Tuple(
                    (
                        spaces.Box(
                            low=-1,
                            high=1,
                            shape=(258,),
                        ),
                    )
                    * env_config.get("n_agents"))
                )
                #"state": spaces.Box(
                    #low=0.0, high=1.0, shape=self.cfg["world_shape"] + [2]
                #),
            }
        )

        agent_action_space = spaces.Discrete(12)
        self.action_space = spaces.Tuple((agent_action_space,) * 2)


        self.action_space = spaces.Tuple((agent_action_space,) * self.cfg["n_agents"])

        self.agents = [Agent(i,range_observation = 4)  for i in range(env_config.get('n_agents'))]

        self.reset()

    def seed(self, seed=None):
        self.random_state, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        reset_actions = [agent.reset() for agent in self.agents]
        self.goal_poses = [agent.goal for agent in self.agents]
        self.timestep = 0
        return self.step(reset_actions)[0]

    def step(self, actions):
        actions_white = actions[0]
        actions_black = actions[1]
        
        self.timestep += 1
        

        observations = [
            agent.step(action) for agent, action in zip(self.agents, actions)
        ]
        #take  out agent class

        rewards = {}
        # shift each agent's goal so that the shared NN has to be used to solve the problem
        shifted_poses = (
            self.goal_poses[self.cfg["goal_shift"] :]
            + self.goal_poses[: self.cfg["goal_shift"]]
        )
        for i, (agent, goal) in enumerate(zip(self.agents, shifted_poses)):
            rewards[i] = -1 if not agent.reached_goal else 0
            if not agent.reached_goal and np.linalg.norm(agent.pose - goal) < 1:
                rewards[i] = 1
                agent.reached_goal = True

        all_reached_goal = all([agent.reached_goal for agent in self.agents])
        max_timestep_reached = self.timestep == self.cfg["max_episode_len"]
        done = all_reached_goal or max_timestep_reached

        global_state = np.zeros(self.cfg["world_shape"] + [2], dtype=np.uint8)
   
        for agent in self.agents:
            global_state[int(agent.pose[Y]), int(agent.pose[X]), 0] = 1
            global_state[int(agent.goal[Y]), int(agent.goal[X]), 1] = 1

        obs = {"agents": tuple(observations)}
        
        info = {"rewards": rewards}
        all_rewards = sum(rewards.values())

        return obs, all_rewards, done, info

   
