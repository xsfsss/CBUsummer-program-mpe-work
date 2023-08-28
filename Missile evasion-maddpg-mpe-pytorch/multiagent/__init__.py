import os
import warnings

from gym.envs.registration import register

# Multiagent envs
# ----------------------------------------

register(
    id='MultiagentSimple-v0',
    entry_point='multiagent.envs:SimpleEnv',
    # FIXME(cathywu) currently has to be exactly max_path_length parameters in
    # rllab run script
    max_episode_steps=100,
)

register(
    id='MultiagentSimpleSpeakerListener-v0',
    entry_point='multiagent.envs:SimpleSpeakerListenerEnv',
    max_episode_steps=100,
)

warnings.warn("to results run on this version of the environments. \n")

if os.getenv('SUPPRESS_MA_PROMPT') != '1':
    input("Please read the raised warning, then press Enter to continue... (to suppress this prompt, please set the environment variable `SUPPRESS_MA_PROMPT=1`)\n")
