from gym.envs.registration import register
register(
    id='VirtualHome-v1',
    entry_point='virtual_home.envs.graph_environment_v1:GraphEnvironment',
)

register(
    id='VirtualHome-v2',
    entry_point='virtual_home.envs.graph_environment_v2:GraphEnvironment',
)
