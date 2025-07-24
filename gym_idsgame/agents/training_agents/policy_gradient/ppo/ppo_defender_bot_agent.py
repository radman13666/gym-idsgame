"""
A bot defender agent for the gym-idsgame environment that acts greedily according to a pre-trained policy network
"""
from typing import Union
import numpy as np
import torch
from torch.distributions import Categorical
import traceback
from sklearn import preprocessing
from gym_idsgame.agents.bot_agents.bot_agent import BotAgent
from gym_idsgame.agents.training_agents.models.fnn_actor_critic import FFNActorCritic
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent import PolicyGradientAgent
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.agents.training_agents.openai_baselines.common.ppo.ppo import PPO
from gym_idsgame.envs.idsgame_env import IdsGameEnv
import gym_idsgame.envs.util.idsgame_util as util

class PPODefenderBotAgent(BotAgent):
    """
    Class implementing an defense policy that acts greedily according to a given policy network
    """

    def __init__(self, pg_config: PolicyGradientAgentConfig, game_config: GameConfig, model_path: str = None,
                 env: IdsGameEnv = None):
        """
        Constructor, initializes the policy

        :param game_config: the game configuration
        """
        super(PPODefenderBotAgent, self).__init__(game_config)
        if model_path is None:
            raise ValueError("Cannot create a PPODefenderBotAgent without specifying the path to the model")
        self.idsgame_env = env
        self.config = pg_config
        self.model_path = model_path
        self.initialize_models()
        self.device = "cpu" if not self.config.gpu else "cuda:" + str(self.config.gpu_id)


    def initialize_models(self) -> None:
        """
        Initialize models
        :return: None
        """

        # Initialize models
        self.defender_policy_network = FFNActorCritic(
            self.config.input_dim_attacker, 
            self.config.output_dim_attacker,
            self.config.hidden_dim, 
            num_hidden_layers=self.config.num_hidden_layers, hidden_activation=self.config.hidden_activation
        )

        self.defender_policy_network.to(self.device)

        self.defender_policy_network.load_state_dict(torch.load(self.model_path))
        self.defender_policy_network.eval()

    def action(self, game_state: GameState) -> int:
        """
        Samples an action from the policy.

        :param game_state: the game state
        :return: action_id
        """
        # Feature engineering
        attacker_obs = game_state.get_attacker_observation(
            self.game_config.network_config, 
            local_view=self.idsgame_env.local_view_features(),
            reconnaissance=self.game_config.reconnaissance_actions,
            reconnaissance_bool_features=self.idsgame_env.idsgame_config.reconnaissance_bool_features
        )
        defender_obs = game_state.get_defender_observation(self.game_config.network_config)
        defender_state = self.update_state(
            attacker_obs=attacker_obs, 
            defender_obs=defender_obs, state=[]
        )

        # get attacker agent action
        defender_action = self.get_action(defender_state)

        return defender_action

    def get_action(self, state: np.ndarray) -> int:
        """
        Samples an action from the policy network

        :param state: the state to sample an action for
        
        :return: The sampled action id
        """
        # print(f'state: {state}')

        state = torch.from_numpy(state.flatten()).float()

        # Move to GPU if using GPU
        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:" + str(self.config.gpu_id))
            state = state.to(device)

        # Calculate legal actions
        actions = list(range(self.idsgame_env.num_defense_actions))
        legal_actions = list(filter(lambda action: self.idsgame_env.is_defense_legal(action), actions))
        non_legal_actions = list(filter(lambda action: not self.idsgame_env.is_defense_legal(action), actions))
        
        # Forward pass using the current policy network to predict P(a|s)
        action_probs, state_value = self.defender_policy_network(state)
        
        # Set probability of non-legal actions to 0
        action_probs_1 = action_probs.clone()
        if len(legal_actions) > 0:
            action_probs_1[non_legal_actions] = 0

        # Use torch.distributions package to create a parameterizable probability distribution of the learned policy
        # PG uses a trick to turn the gradient into a stochastic gradient which we can sample from in order to
        # approximate the true gradient (which we can’t compute directly). It can be seen as an alternative to the
        # reparameterization trick
        policy_dist = Categorical(action_probs_1)

        # Sample an action from the probability distribution
        action = policy_dist.sample()

        # log_prob returns the log of the probability density/mass function evaluated at value.
        # save the log_prob as it will use later on for computing the policy gradient
        # policy gradient theorem says that the stochastic gradient of the expected return of the current policy is
        # the log gradient of the policy times the expected return, therefore we save the log of the policy distribution
        # now and use it later to compute the gradient once the episode has finished.
        log_prob = policy_dist.log_prob(action)

        return action.item()#, log_prob, state_value, action_probs

    def update_state(self, attacker_obs: np.ndarray = None, defender_obs: np.ndarray = None, state: np.ndarray = None) -> np.ndarray:
        """
        Update approximative Markov state

        :param attacker_obs: attacker obs
        :param defender_obs: defender observation
        :param state: current state
        
        :return: new state
        """
        if self.env.fully_observed():
            if self.config.merged_ad_features:
                a_pos = attacker_obs[:,-1]
                det_values = defender_obs[:, -1]
                temp = defender_obs[:,0:-1] - attacker_obs[:,0:-1]
                if self.config.normalize_features:
                    det_values = det_values / np.linalg.norm(det_values)
                    temp = temp / np.linalg.norm(temp)
                features = []
                for idx, row in enumerate(temp):
                    t = row.tolist()
                    t.append(a_pos[idx])
                    t.append(det_values[idx])
                    features.append(t)
                features = np.array(features)
                if self.config.state_length == 1:
                    return features
                if len(state) == 0:
                    s = np.array([features] * self.config.state_length)
                    return s
                state = np.append(state[1:], np.array([features]), axis=0)
            else:
                if self.config.state_length == 1:
                    return np.append(attacker_obs, defender_obs)
                if len(state) == 0:
                    temp = np.append(attacker_obs, defender_obs)
                    s = np.array([temp] * self.config.state_length)
                    return s
                temp = np.append(attacker_obs, defender_obs)
                state = np.append(state[1:], np.array([temp]), axis=0)
            return state
        else:
            if self.config.normalize_features:
                attacker_obs_1 = attacker_obs[:,0:-1] / np.linalg.norm(attacker_obs[:,0:-1])
                normalized_attacker_features = []
                for idx, row in enumerate(attacker_obs_1):
                    if np.isnan(attacker_obs_1).any():
                        t = attacker_obs[idx]
                    else:
                        t = attacker_obs_1.tolist()
                        t.append(attacker_obs[idx][-1])
                    normalized_attacker_features.append(t)

                defender_obs_1 = defender_obs[:, 0:-1] / np.linalg.norm(defender_obs[:, 0:-1])
                normalized_defender_features = []
                for idx, row in enumerate(defender_obs_1):
                    if np.isnan(defender_obs_1).any():
                        t= defender_obs[idx]
                    else:
                        t = defender_obs_1.tolist()
                        t.append(defender_obs[idx][-1])
                    normalized_defender_features.append(t)
                attacker_obs = np.array(normalized_attacker_features)
                defender_obs = np.array(normalized_defender_features)

            if self.config.state_length == 1:
                return np.array(defender_obs)
            
            if len(state) == 0:
                return np.array([defender_obs] * self.config.state_length)
            
            state = np.append(state[1:], np.array([defender_obs]), axis=0)
            return state

    def grid_obs(self, attacker_obs, defender_obs, attacker=True):
        if attacker and self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
            a_obs_len = self.idsgame_env.idsgame_config.game_config.num_attack_types + 1
            defender_obs = attacker_obs[:, a_obs_len:]
            attacker_obs = attacker_obs[:, 0:a_obs_len]
            attacker_position = attacker_obs[:, -1]
            attacker_obs = attacker_obs[:, 0:-1]
        elif self.idsgame_env.fully_observed():
            a_obs_len = self.idsgame_env.idsgame_config.game_config.num_attack_types + 1
            attacker_position = attacker_obs[:, -1]
            attacker_obs = attacker_obs[:, 0:-1]
            defender_obs = defender_obs[:, 0:-1]

        attack_plane = attacker_obs
        if self.config.normalize_features:
            normalized_attack_plane = preprocessing.normalize(attack_plane)

        defense_plane = defender_obs
        if self.config.normalize_features:
            normalized_defense_plane = preprocessing.normalize(defense_plane)

        position_plane = np.zeros(attack_plane.shape)
        for idx, present in enumerate(attacker_position):
            position_plane[idx] = np.full(position_plane.shape[1], present)

        reachable_plane = np.zeros(attack_plane.shape)
        attacker_row, attacker_col = self.idsgame_env.state.attacker_pos
        attacker_matrix_id = self.idsgame_env.idsgame_config.game_config.network_config.get_adjacency_matrix_id(
            attacker_row, attacker_col)
        for node_id in range(len(attack_plane)):
            node_row, node_col = self.idsgame_env.idsgame_config.game_config.network_config.get_node_pos(node_id)
            adj_matrix_id = self.idsgame_env.idsgame_config.game_config.network_config.get_adjacency_matrix_id(node_row,
                                                                                                               node_col)
            reachable = self.idsgame_env.idsgame_config.game_config.network_config.adjacency_matrix[attacker_matrix_id][
                            adj_matrix_id] == int(1)
            if reachable:
                val = 1
            else:
                val = 0
            reachable_plane[node_id] = np.full(reachable_plane.shape[1], val)

        row_difference_plane = np.zeros(attack_plane.shape)
        for node_id in range(len(attack_plane)):
            node_row, node_col = self.idsgame_env.idsgame_config.game_config.network_config.get_node_pos(node_id)
            row_difference = attacker_row - node_row
            row_difference_plane[node_id] = np.full(row_difference_plane.shape[1], row_difference)

        if self.config.normalize_features:
            normalized_row_difference_plance = preprocessing.normalize(row_difference_plane)

        attack_defense_difference_plane = attacker_obs - defender_obs
        if self.config.normalize_features:
            normalized_attack_defense_difference_plane = preprocessing.normalize(attack_defense_difference_plane)

        if self.config.normalize_features:
            feature_frames = np.stack(
                [normalized_attack_plane, normalized_defense_plane, position_plane, reachable_plane,
                 normalized_row_difference_plance,
                 normalized_attack_defense_difference_plane],
                axis=0)
        else:
            feature_frames = np.stack(
                [attack_plane, defense_plane, position_plane, reachable_plane,
                 row_difference_plane,
                 attack_defense_difference_plane],
                axis=0)
        # print("feature_frames:")
        # print(feature_frames)
        # raise AssertionError("test")
        return feature_frames

    def create_policy_plot(self, distribution, episode, attacker=True) -> None:
        """
        Utility function for creating a density plot of the policy distribution p(a|s) and add to Tensorboard

        :param distribution: the distribution to plot
        :param episode: the episode when the distribution was recorded
        :param attacker: boolean flag whether it is the attacker or defender
        :return: None
        """
        #distribution = distribution/np.linalg.norm(distribution, ord=np.inf, axis=0, keepdims=True)
        distribution /= np.sum(distribution)
        sample = np.random.choice(list(range(len(distribution))), size=1000, p=distribution)
        tag = "Attacker"
        file_suffix = "initial_state_policy_attacker"
        if not attacker:
            tag = "Defender"
            file_suffix = "initial_state_policy_defender"
        title = tag + " Initial State Policy"
        data = util.action_dist_hist(sample, title=title, xlabel="Action", ylabel=r"$\mathbb{P}(a|s)$",
                                             file_name=self.config.save_dir + "/" + file_suffix + "_" + str(episode))
        # self.tensorboard_writer.add_image(str(episode) + "_initial_state_policy/" + tag,
        #                                   data, global_step=episode, dataformats="HWC")
