import argparse
import json
import os
import shutil, errno
from datetime import datetime, timedelta

import pathlib
import genesis as gs

from vico.env import VicoEnv
from vico.modules import *
from vico.tools.constants import ASSETS_PATH
from vico.tools.utils import get_height_at, load_height_field


# Helper function to generate target_goods
def generate_target_goods(num_members):
	total = 2 * num_members
	# Start with a list of ones to ensure each index has at least 1
	goods = [1] * 4
	remaining = total - 4  # Subtract the initial 1s from the total
	# Distribute the remaining randomly across the 4 indices
	for _ in range(remaining):
		goods[random.randint(0, 3)] += 1
	return goods


def evaluate(data_dir):
	from evaluate import evaluate_LQ
	evaluate_LQ(data_dir)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--precision", type=str, default='32')
	parser.add_argument("--logging_level", type=str, default='info')
	parser.add_argument("--backend", type=str, default='gpu')
	parser.add_argument("--head_less", '-l', action='store_true')
	parser.add_argument("--multi_process", '-m', action='store_true')
	parser.add_argument("--model_server_port", type=int, default=0)
	parser.add_argument("--output_dir", "-o", type=str, default='output')
	parser.add_argument("--debug", action='store_true')
	parser.add_argument("--overwrite", action='store_true')
	parser.add_argument("--challenge", type=str, default='full')

	### Simulation configurations
	parser.add_argument("--resolution", type=int, default=512)
	parser.add_argument("--enable_collision", action='store_true')
	parser.add_argument("--enable_decompose", action='store_true')
	parser.add_argument("--skip_avatar_animation", action='store_true')
	parser.add_argument("--enable_gt_segmentation", action='store_true')
	parser.add_argument("--max_seconds", type=int, default=86400) # 24 hours
	parser.add_argument("--save_per_seconds", type=int, default=10)
	parser.add_argument("--enable_third_person_cameras", action='store_true')
	parser.add_argument("--enable_demo_camera", action='store_true')
	parser.add_argument("--batch_renderer", action='store_true')
	parser.add_argument("--curr_time", type=str)

	### Scene configurations
	parser.add_argument("--scene", type=str, default='NY')
	parser.add_argument("--no_load_indoor_scene", action='store_true')
	parser.add_argument("--no_load_indoor_objects", action='store_true')
	parser.add_argument("--no_load_outdoor_objects", action='store_true')
	parser.add_argument("--outdoor_objects_max_num", type=int, default=10)
	parser.add_argument("--no_load_scene", action='store_true')

	# Traffic configurations
	parser.add_argument("--no_traffic_manager", action='store_true')
	parser.add_argument("--tm_vehicle_num", type=int, default=0)
	parser.add_argument("--tm_avatar_num", type=int, default=0)
	parser.add_argument("--enable_tm_debug", action='store_true')

	### Agent configurations
	parser.add_argument("--num_agents", type=int, default=15)
	parser.add_argument("--config", type=str, default='agents_num_15')
	parser.add_argument("--agent_type", type=str, default='ella')
	parser.add_argument("--detect_interval", type=int, default=1)
	parser.add_argument("--region_layer", action='store_true')

	parser.add_argument("--lm_source", type=str, choices=["openai", "azure", "huggingface", "local"]
						, default="azure", help="language model source")
	parser.add_argument("--lm_id", "-lm", type=str, default="gpt-4o", help="language model id")

	args = parser.parse_args()

	# Set up environment
	args.output_dir = os.path.join(args.output_dir, f"{args.scene}_{args.config}", f"{args.agent_type}")
	if args.overwrite and os.path.exists(args.output_dir):
		print(f"Overwriting output directory: {args.output_dir}")
		shutil.rmtree(args.output_dir)
	os.makedirs(args.output_dir, exist_ok=True)

	config_path = os.path.join(args.output_dir, 'curr_sim')
	continued = False
	if not os.path.exists(config_path):
		seed_config_path = os.path.join(ASSETS_PATH, 'scenes', args.scene, args.config)
		print(f"Initializing new simulation from config: {seed_config_path}")
		try:
			shutil.copytree(seed_config_path, config_path)
		except OSError as exc:  # python >2.5
			if exc.errno in (errno.ENOTDIR, errno.EINVAL):
				shutil.copy(seed_config_path, config_path)
			else:
				raise
	else:
		continued = True
		print(f"Continuing simulation from config: {config_path}")
	
	config = json.load(open(os.path.join(config_path, 'config.json'), 'r'))
	if 'group_task_info' in config:
		continued = True
		print(f"Continuing Controlled Finals Leadership Quest simulation from config: {config_path}")
	else:
		continued = False
		print(f"Initializing new Controlled Finals Leadership Quest simulation from config: {config_path}, setting up group leaders...")
		start_date = datetime.strptime(config['start_time'], "%B %d, %Y, %H:%M:%S").date()
		curr_date = start_date + timedelta(days=1)
		if args.curr_time is None:
			args.curr_time = "09:00:00"
		config['curr_time'] = f"{curr_date.strftime('%B %d, %Y')}, {args.curr_time}"

		groups_data = config['groups']
		place_metadata = json.load(open(os.path.join(config_path, 'place_metadata.json'),'r'))
		building_metadata = json.load(open(os.path.join(config_path, 'building_metadata.json'),'r'))
		terrain_height_field = load_height_field(os.path.join(gs.utils.get_assets_dir(), "ViCo/scenes/v1", args.scene, "height_field.npz"))
		goods_categories = ['Fresh Produce', 'Dairy & Meat', 'Beverages', 'Snacks']
		random.seed(args.seed)

		# Generate the updated dictionary
		group_task_info = {
			"goods_categories": goods_categories,  # Define the goods categories at the top level
			**{
				group_name: {
					"group_leader": random.choice(group_info["members"]),
					"target_goods": generate_target_goods(len(group_info["members"])),
					"achieved_goods": [0,0,0,0],
				}
				for group_name, group_info in groups_data.items()
			}
		}

		config['group_task_info'] = group_task_info
		
		# set daily requirements
		initial_agent_pos = []
		num_agents = len(config["agent_names"])
		coarse_indoor_scene = json.load(open(os.path.join(ASSETS_PATH, "coarse_type_to_indoor_scene.json"), 'r'))
		room_agents = {}
		for i, agent_name in enumerate(config["agent_names"]):
			agent_scratch_path = os.path.join(config_path, agent_name, "scratch.json")
			agent_scratch = json.load(open(agent_scratch_path, 'r'))

			group = agent_scratch['groups'][0]
			group_name = group['name']
			group_place = group['place']
			group_building = place_metadata[group_place]['building']

			# randomly spawn agents in the room and facing each other
			place = place_metadata[group_place]
			indoor_x, indoor_y, indoor_z = place['location']
			if group_building != "open space" and 'scene' in place:
				indoor_scene = json.load(open(os.path.join(ASSETS_PATH, place['scene']), 'r'))
				if indoor_scene['type'] == 'glb':
					import math
					indoor_x += math.cos(i * 2 * math.pi / num_agents) * 2 + indoor_scene['left'] + indoor_scene['width'] / 2
					indoor_y += math.sin(i * 2 * math.pi / num_agents) * 2 + indoor_scene['top'] + indoor_scene['height'] / 2
					config['agent_poses'][i] = [indoor_x, indoor_y, indoor_z, 0.0, 0.0, i * 2 * math.pi / num_agents + math.pi]
				else:
					offset = np.array(place['location'])
					cnt = room_agents.get(indoor_scene['name'], 0)
					room_agents[indoor_scene['name']] = cnt + 1
					pose = indoor_scene.get("group_avatar_pos", [])[cnt]
					pos, euler = pose['pos'], pose['euler']
					config['agent_poses'][i] = np.array([pos[0] + offset[0], pos[1] + offset[1], offset[2], euler[0], euler[1], euler[2]])

			if group_building == "open space":
				outdoor_pose = [indoor_x, indoor_y, indoor_z, 0.0, 0.0, 0.0]
			else:
				outdoor_x, outdoor_y = building_metadata[group_building]["outdoor_xy"]
				outdoor_z = get_height_at(terrain_height_field, outdoor_x, outdoor_y)
				# print(f"Agent {config['agent_names'][i]}'s new outdoor pose: {outdoor_x}, {outdoor_y}, {outdoor_z}")
				outdoor_pose = [outdoor_x, outdoor_y, outdoor_z, 0.0, 0.0, 0.0]

			agent_scratch['current_place'] = group_place

			config["agent_infos"][i].update({
				"outdoor_pose": outdoor_pose,
				"current_building": group_building,
				"current_place": group_place,
				"cash": 50,
			})

			# Generate the target goods string
			target_goods_string = ", ".join(
				f"{quantity} {category}" for category, quantity in 
				zip(goods_categories, group_task_info[group_name]['target_goods'])
			)
			
			if agent_name in group_task_info[group_name]['group_leader']:
				members_str = ", ".join(m for m in group['members'] if m != agent_name)
				agent_scratch["daily_requirement"] = f"I am the leader of my group {group_name}. I need to discuss and assign tasks to my group members to collect {target_goods_string} from stores and bring them back here at {group_place} before 12:00:00. Each person could take 2 items at a time with each of his hand."
			else:
				leader = group_task_info[group_name]['group_leader']					
				agent_scratch["daily_requirement"] = f"I need to help my group {group_name}'s leader {leader} to prepare for a group activity. I will discuss with my leader about the items to collect, follow the instructions given by my leader and complete my assigned task before 12:00:00."
			
			agent_scratch["currently"] = agent_scratch["daily_requirement"]
			with open(agent_scratch_path, 'w') as file:
				json.dump(agent_scratch, file, indent=4, default=json_converter)

		with open(os.path.join(config_path, 'config.json'), 'w') as file:
			json.dump(config, file, indent=4, default=json_converter)


	os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

	from tools.model_manager import global_model_manager
	global_model_manager.init(local=True)
	
	# Initialize environment
	env = VicoEnv(
		seed=args.seed,
		precision=args.precision,
		logging_level=args.logging_level,
		backend= gs.cpu if args.backend == 'cpu' else gs.gpu,
		head_less=args.head_less,
		resolution=args.resolution,
		challenge="leadership_quest",
		num_agents=args.num_agents,
		config_path=config_path,
		scene=args.scene,
		enable_indoor_scene=not args.no_load_indoor_scene,
		enable_indoor_objects=not args.no_load_indoor_objects,
		enable_outdoor_objects=not args.no_load_outdoor_objects,
		outdoor_objects_max_num=args.outdoor_objects_max_num,
		enable_collision=args.enable_collision,
		enable_decompose=args.enable_decompose,
		skip_avatar_animation=args.skip_avatar_animation,
		enable_gt_segmentation=args.enable_gt_segmentation,
		no_load_scene=args.no_load_scene,
		output_dir=args.output_dir,
		enable_third_person_cameras=args.enable_third_person_cameras,
		enable_demo_camera=args.enable_demo_camera,
		no_traffic_manager=args.no_traffic_manager,
		enable_tm_debug=args.enable_tm_debug,
		tm_vehicle_num=args.tm_vehicle_num,
		tm_avatar_num=args.tm_avatar_num,
		save_per_seconds=args.save_per_seconds,
		defer_chat="ella" in args.agent_type,
		debug=args.debug,
		batch_renderer=args.batch_renderer,
	)

	agents = []
	
	from agents import get_agent_cls, AgentProcess
	for i in range(args.num_agents):
		basic_kwargs = dict(
			name = env.agent_names[i],
			pose = env.config["agent_poses"][i],
			info = env.agent_infos[i],
			sim_path = config_path,
			no_react = args.no_react,
			debug = args.debug,
			logging_level = args.logging_level,
			multi_process = args.multi_process,
		)
		llm_kwargs = dict(
			lm_source=args.lm_source,
			lm_id=args.lm_id,
		)

		agent_cls = get_agent_cls(agent_type=args.agent_type)
		agents.append(AgentProcess(agent_cls, **basic_kwargs, **llm_kwargs))


	if args.multi_process:
		gs.logger.info("Start agent processes")
		for agent in agents:
			agent.start()
		gs.logger.info("Agent processes started")


	# Simulation loop
	obs = env.reset()
	agent_list_to_update = obs.pop('agent_list_to_update')
	agent_actions = {}
	agent_actions_to_print = {}
	args.max_steps = args.max_seconds // env.sec_per_step
	while True:
		lst_time = time.perf_counter()
		for i, agent in enumerate(agents):
			agent.update(obs[i])
		for i, agent in enumerate(agents):
			agent_actions[i] = agent.act()
			agent_actions_to_print[agent.name] = agent_actions[i]['type'] if agent_actions[i] is not None else None
			if agent_actions[i] is not None and agent_actions[i]['type'] == 'converse':
				agent_actions[i]['request_chat_func'] = agent.request_chat
				agent_actions[i]['get_utterance_func'] = agent.get_utterance
		agent_actions['agent_list_to_update'] = agent_list_to_update
		gs.logger.info(f"current time: {env.curr_time}, ViCo steps: {env.steps}, agents actions: {agent_actions_to_print}")
		sps_agent = time.perf_counter() - lst_time
		env.config["sps_agent"] = (env.config["sps_agent"] * env.steps + sps_agent) / (env.steps + 1)
		lst_time = time.perf_counter()
		obs, _, done, info = env.step(agent_actions)
		agent_list_to_update = obs.pop('agent_list_to_update')
		sps_sim = time.perf_counter() - lst_time
		env.config["sps_sim"] = (env.config["sps_sim"] * (env.steps - 1) + sps_sim) / max(env.steps, 1)
		gs.logger.info(f"Time used: {sps_agent:.2f}s for agents, {sps_sim:.2f}s for simulation, "
					   f"average {env.config['sps_agent']:.2f}s for agents, "
					   f"{env.config['sps_sim']:.2f}s for simulation, "
					   f"{env.config['sps_chat']:.2f}s for post-chatting over {env.steps} steps.")
		if env.steps > args.max_steps:
			break

	evaluate(args.output_dir)

	for agent in agents:
		agent.close()
	env.close()