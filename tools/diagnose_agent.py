import os
from agents.ella import EllaAgent
from tools.model_manager import global_model_manager

if __name__ == "__main__":
	global_model_manager.init(local=True)
	agent = EllaAgent(name="Emily Johnson", pose=[980.8406653404236,
            957.0423936843872,
            -4.0,], info={'cash': 100, 'held_objects': [None, None]}, sim_path="output/odm/DETROIT_agents_num_15/ella/curr_sim", debug=True)
	print(agent.s_mem.get_position_from_name("dresser_0"))
    # agent.last_react_time = agent.curr_time
	# print(agent._act({'rgb': None, 'curr_time': agent.curr_time}))
	global_model_manager.close()