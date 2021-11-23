from util.rlkit_custom import rollout

from rlkit.torch.pytorch_util import set_gpu_mode

import torch
from torch import nn as nn
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
# from rlkit.torch.sac.sac import SACTrainer
from util.old_masac import MASACTrainer
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from util.rlkit_custom import CustomTorchBatchRLAlgorithm

from rlkit.core import logger
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper

from robosuite.controllers import load_controller_config, ALL_CONTROLLERS

import numpy as np

#==================== Custom policy for multiagent case =================================
#
#
# from rlkit.policies.base import ExplorationPolicy, Policy
# from rlkit.torch.core import eval_np
# from rlkit.torch.distributions import TanhNormal
# from rlkit.torch.networks import Mlp
#
# LOG_SIG_MAX = 2
# LOG_SIG_MIN = -20
#
# class MATanhGaussianPolicy(Mlp, ExplorationPolicy):
#     def __init__(
#             self,
#             hidden_sizes,
#             obs_dim,
#             action_dim,
#             std=None,
#             init_w=1e-3,
#             **kwargs
#     ):
#         super().__init__(
#             hidden_sizes,
#             input_size=obs_dim,
#             output_size=action_dim,
#             init_w=init_w,
#             **kwargs
#         )
#         self.log_std = None
#         self.std = std
#
#         # self.qf1_agent0 = qf1_agent0
#
#         if std is None:
#             last_hidden_size = obs_dim
#             if len(hidden_sizes) > 0:
#                 last_hidden_size = hidden_sizes[-1]
#             self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
#             self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
#             self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
#         else:
#             self.log_std = np.log(std)
#             assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX
#
#     def forward(self, obs):
#
#         h = obs
#         for i, fc in enumerate(self.fcs):
#             h = self.hidden_activation(fc(h))
#         # mean = self.last_fc(h)
#
#         mean1 = self.last_fc(h)
#         mean2 = self.last_fc(h)
#
#         mean = torch.stack((mean1, mean2), axis=0)
#
#         if self.std is None:
#             log_std = self.last_fc_log_std(h)
#             log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
#             std1 = torch.exp(log_std)
#             std2 = torch.exp(log_std)
#         else:
#             std1 = torch.from_numpy(np.array([self.std, ])).float().to(
#                 ptu.device)
#             std2 = torch.from_numpy(np.array([self.std, ])).float().to(
#                 ptu.device)
#
#         std = torch.stack((std1, std2), axis=0)
#
#         return TanhNormal(mean, std)
#
#     # def get_action(self, obs_np, deterministic=False):
#     #     actions = self.get_actions(obs_np[None], deterministic=deterministic)
#     #     return actions[0, :], {}
#     #
#     # def get_actions(self, obs_np, deterministic=False):
#     #     return eval_np(self, obs_np, deterministic=deterministic)[0]
#
#     def get_action(self, obs_np):
#         actions = self.get_actions(obs_np[None])
#         return actions[0, :], {}
#
#     def get_actions(self, obs_np):
#         return eval_np(self, obs_np)


import numpy as np
import torch
from torch import nn as nn

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.core import eval_np
# from rlkit.torch.distributions import TanhNormal
# from util.distributions_multi import MultiTanhNormal
from util.distributions import TanhNormal
from rlkit.torch.networks import Mlp
from collections import namedtuple

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class MATanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:
    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.
    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.agent0_policy = TanhGaussianPolicy(hidden_sizes=hidden_sizes, obs_dim=obs_dim, action_dim=action_dim//2, std=std, init_w=init_w, **kwargs)
        self.agent1_policy = TanhGaussianPolicy(hidden_sizes=hidden_sizes, obs_dim=obs_dim, action_dim=action_dim//2, std=std, init_w=init_w, **kwargs)

        # self.log_std = None
        # self.std = std
        # if std is None:
        #     last_hidden_size = obs_dim
        #     if len(hidden_sizes) > 0:
        #         last_hidden_size = hidden_sizes[-1]
        #     self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
        #     self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
        #     self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        # else:
        #     self.log_std = np.log(std)
        #     assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        action_0, mean_0, log_std_0, log_prob_0, entropy_0, std_0, mean_action_log_prob_0, pre_tanh_value_0 = self.agent0_policy(obs, reparameterize=reparameterize, deterministic=deterministic, return_log_prob=return_log_prob)
        action_1, mean_1, log_std_1, log_prob_1, entropy_1, std_1, mean_action_log_prob_1, pre_tanh_value_1 = self.agent0_policy(obs, reparameterize=reparameterize, deterministic=deterministic, return_log_prob=return_log_prob)

        action = torch.cat((action_0, action_1), axis=-1)
        mean = torch.cat((mean_0, mean_1), axis=-1)
        log_std = torch.cat((log_std_0, log_std_1), axis=-1)

        if log_prob_0 is None:
            log_prob = None
        else:
            log_prob = torch.cat((log_prob_0, log_prob_1), axis=-1)
        if entropy_0 is None:
            entropy = None
        else:
            entropy = torch.cat((entropy_0, entropy_1), axis=-1)

        std = torch.cat((std_0, std_1), axis=-1)

        if mean_action_log_prob_0 is None:
            mean_action_log_prob = None
        else:
            mean_action_log_prob = torch.cat((mean_action_log_prob_0, mean_action_log_prob_1), axis=-1)
        if pre_tanh_value_0 is None:
            pre_tanh_value = None
        else:
            pre_tanh_value = torch.cat((pre_tanh_value_0, pre_tanh_value_1), axis=-1)
        # return ()
        # h = obs
        # for i, fc in enumerate(self.fcs):
        #     h = self.hidden_activation(fc(h))
        #
        # mean1 = self.last_fc(h)
        # mean2 = self.last_fc(h)
        # import pdb; pdb.set_trace()
        # mean = torch.stack((mean1, mean2), axis=-1)
        #
        # if self.std is None:
        #     log_std = self.last_fc_log_std(h)
        #     log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        #     std1 = torch.exp(log_std)
        #     std2 = torch.exp(log_std)
        # else:
        #     std1 = self.std
        #     std2 = self.std
        #     log_std = self.log_std
        # std = torch.stack((std1, std2), axis=0)
        #
        # log_prob = None
        # entropy = None
        # mean_action_log_prob = None
        # pre_tanh_value = None
        # if deterministic:
        #     action = torch.tanh(mean)
        # else:
        #     tanh_normal = TanhNormal(mean, std)
        #     if return_log_prob:
        #         if reparameterize is True:
        #             action, pre_tanh_value = tanh_normal.rsample(
        #                 return_pretanh_value=True
        #             )
        #         else:
        #             action, pre_tanh_value = tanh_normal.sample(
        #                 return_pretanh_value=True
        #             )
        #         log_prob = tanh_normal.log_prob(
        #             action,
        #             pre_tanh_value=pre_tanh_value
        #         )
        #         log_prob = log_prob.sum(dim=1, keepdim=True)
        #     else:
        #         if reparameterize is True:
        #             action = tanh_normal.rsample()
        #         else:
        #             action = tanh_normal.sample()
        # import pdb; pdb.set_trace()
        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )

#======================================================================================

# Define agents available
# AGENTS = {"SAC", "TD3", "MASAC"}
AGENTS = {"MASAC"}

def experiment(variant, agent="MASAC"):

    # Make sure agent is a valid choice
    assert agent in AGENTS, "Invalid agent selected. Selected: {}. Valid options: {}".format(agent, AGENTS)

    # Get environment configs for expl and eval envs and create the appropriate envs
    # suites[0] is expl and suites[1] is eval
    suites = []
    for env_config in (variant["expl_environment_kwargs"], variant["eval_environment_kwargs"]):
        # Load controller
        controller = env_config.pop("controller")
        if controller in set(ALL_CONTROLLERS):
            # This is a default controller
            controller_config = load_controller_config(default_controller=controller)
        else:
            # This is a string to the custom controller
            controller_config = load_controller_config(custom_fpath=controller)
        # Create robosuite env and append to our list
        suites.append(suite.make(**env_config,
                                 has_renderer=False,
                                 has_offscreen_renderer=False,
                                 use_object_obs=True,
                                 use_camera_obs=False,
                                 reward_shaping=True,
                                 controller_configs=controller_config,
                                 ))
    # Create gym-compatible envs
    expl_env = NormalizedBoxEnv(GymWrapper(suites[0]))
    eval_env = NormalizedBoxEnv(GymWrapper(suites[1]))

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    qf1_agent0 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )

    qf1_agent1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )

    qf2_agent0 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )

    qf2_agent1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )

    target_qf1_agent0 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )

    target_qf1_agent1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )

    target_qf2_agent0 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )

    target_qf2_agent1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )

    # Define references to variables that are agent-specific
    trainer = None
    eval_policy = None
    expl_policy = None

    # Instantiate trainer with appropriate agent
    if agent == "SAC":
        expl_policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **variant['policy_kwargs'],
        )
        eval_policy = MakeDeterministic(expl_policy)
        trainer = SACTrainer(
            env=eval_env,
            policy=expl_policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **variant['trainer_kwargs']
        )
    elif agent == "TD3":
        eval_policy = TanhMlpPolicy(
            input_size=obs_dim,
            output_size=action_dim,
            **variant['policy_kwargs']
        )
        target_policy = TanhMlpPolicy(
            input_size=obs_dim,
            output_size=action_dim,
            **variant['policy_kwargs']
        )
        es = GaussianStrategy(
            action_space=expl_env.action_space,
            max_sigma=0.1,
            min_sigma=0.1,  # Constant sigma
        )
        expl_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=es,
            policy=eval_policy,
        )
        trainer = TD3Trainer(
            policy=eval_policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            target_policy=target_policy,
            **variant['trainer_kwargs']
        )
    elif agent == "MASAC":
        # use the custom policy for multi-agent case defined at the beginning
        expl_policy = MATanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **variant['policy_kwargs'],
        )
        eval_policy = MakeDeterministic(expl_policy)
        trainer = MASACTrainer(
            env=eval_env,
            policy=expl_policy,
            qf1_agent0=qf1_agent0,
            qf1_agent1=qf1_agent1,
            qf2_agent0=qf2_agent0,
            qf2_agent1=qf2_agent1,
            target_qf1_agent0=target_qf1_agent0,
            target_qf1_agent1=target_qf1_agent1,
            target_qf2_agent0=target_qf2_agent0,
            target_qf2_agent1=target_qf2_agent1,
            **variant['trainer_kwargs']
        )
    else:
        print("Error: No valid agent chosen!")

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )

    # Define algorithm
    algorithm = CustomTorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


def evaluate_policy(env_config, model_path, n_eval, printout=False):
    if printout:
        print("Loading policy...")

    # Load trained model and corresponding policy
    data = torch.load(model_path)
    policy = data['evaluation/policy']

    if printout:
        print("Policy loaded")

    # Load controller
    controller = env_config.pop("controller")
    if controller in set(ALL_CONTROLLERS):
        # This is a default controller
        controller_config = load_controller_config(default_controller=controller)
    else:
        # This is a string to the custom controller
        controller_config = load_controller_config(custom_fpath=controller)

    # Create robosuite env
    env = suite.make(**env_config,
                     has_renderer=False,
                     has_offscreen_renderer=False,
                     use_object_obs=True,
                     use_camera_obs=False,
                     reward_shaping=True,
                     controller_configs=controller_config
                     )
    env = GymWrapper(env)

    # Use CUDA if available
    if torch.cuda.is_available():
        set_gpu_mode(True)
        policy.cuda() if not isinstance(policy, MakeDeterministic) else policy.stochastic_policy.cuda()

    if printout:
        print("Evaluating policy over {} simulations...".format(n_eval))

    # Create variable to hold rewards to be averaged
    returns = []

    # Loop through simulation n_eval times and take average returns each time
    for i in range(n_eval):
        path = rollout(
            env,
            policy,
            max_path_length=env_config["horizon"],
            render=False,
        )

        # Determine total summed rewards from episode and append to 'returns'
        returns.append(sum(path["rewards"]))

    # Average the episode rewards and return the normalized corresponding value
    return sum(returns) / (env_config["reward_scale"] * n_eval)


def simulate_policy(
        env,
        model_path,
        horizon,
        render=False,
        video_writer=None,
        num_episodes=np.inf,
        printout=False,
        use_gpu=False):
    if printout:
        print("Loading policy...")

    # Load trained model and corresponding policy
    map_location = torch.device("cuda") if use_gpu else torch.device("cpu")
    data = torch.load(model_path, map_location=map_location)
    policy = data['evaluation/policy']

    if printout:
        print("Policy loaded")

    # Use CUDA if available
    if torch.cuda.is_available():
        set_gpu_mode(True)
        policy.cuda() if not isinstance(policy, MakeDeterministic) else policy.stochastic_policy.cuda()

    if printout:
        print("Simulating policy...")

    # Create var to denote how many episodes we're at
    ep = 0

    # Loop through simulation rollouts
    while ep < num_episodes:
        if printout:
            print("Rollout episode {}".format(ep))
        path = rollout(
            env,
            policy,
            max_path_length=horizon,
            render=render,
            video_writer=video_writer,
        )

        # Log diagnostics if supported by env
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()

        # Increment episode count
        ep += 1
