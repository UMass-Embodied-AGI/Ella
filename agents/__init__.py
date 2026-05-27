from vico.agents import AgentProcess

def get_agent_cls(agent_type, robot_type=None):
    if 'ella' in agent_type:
        from .ella import EllaAgent
        return EllaAgent
    if 'generative_agent' in agent_type:
        from .gen_agent import GenAgent
        return GenAgent
    # Fall back to vico's built-in agent types
    from vico.agents import get_agent_cls as vico_get_agent_cls
    return vico_get_agent_cls(agent_type, robot_type=robot_type)
