from vico.agents import AgentProcess

def get_agent_cls_ella(agent_type):
    if agent_type == 'ella':
        from .ella import EllaAgent
        return EllaAgent
    else:
        raise NotImplementedError(f"agent type {agent_type} is not supported")