import argparse
import json
import os
import random
import shutil, errno
from datetime import datetime, timedelta

import genesis as gs

from vico.env import VicoEnv
from vico.modules import *
from vico.tools.constants import ASSETS_PATH

def evaluate(data_dir):
	from evaluate import evaluate_IB
	evaluate_IB(data_dir)


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
		
	# Load configuration
	config = json.load(open(os.path.join(config_path, 'config.json'), 'r'))
	if "party_organizer_groups" in config:
		continued = True
		print(f"Continuing Controlled Finals Influence Battle simulation from config: {config_path}")
	else:
		continued = False
		print(f"Initializing new Controlled Finals Influence Battle simulation from config: {config_path}, setting up party organizer groups...")
		start_date = datetime.strptime(config['start_time'], "%B %d, %Y, %H:%M:%S").date()
		curr_date = start_date + timedelta(days=1)
		if args.curr_time is None:
			args.curr_time = "09:00:00"
		config['curr_time'] = f"{curr_date.strftime('%B %d, %Y')}, {args.curr_time}"
		random.seed(args.seed)
		party_organizer_groups_name = random.sample(list(config['groups'].keys()), 2)

		config['party_organizer_groups'] = party_organizer_groups_name
		with open(os.path.join(config_path, 'config.json'), 'w') as file:
			json.dump(config, file, indent=4, default=json_converter)

		# set daily requirements
		for agent_name in config["agent_names"]:
			agent_scratch_path = os.path.join(config_path, agent_name, "scratch.json")
			agent_scratch = json.load(open(agent_scratch_path, 'r'))
			if agent_scratch['groups'][0]['name'] in party_organizer_groups_name:
				agent_scratch["daily_requirement"] = f"My group {agent_scratch['groups'][0]['name']} is organizing a party at {agent_scratch['groups'][0]['place']} from 14:30:00 to 15:00:00 today. I need to go around the city, find and invite people outside of my group to attend our party today."
				agent_scratch["currently"] = agent_scratch["daily_requirement"]
				with open(agent_scratch_path, 'w') as file:
					json.dump(agent_scratch, file, indent=4, default=json_converter)

	os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

	if args.debug:
		args.enable_third_person_cameras = True
		
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
		challenge="influence_battle",
		num_agents=args.num_agents,
		config_path=config_path,
		scene=args.scene,
		no_load_scene=args.no_load_scene,
		enable_indoor_scene=not args.no_load_indoor_scene,
		enable_indoor_objects=not args.no_load_indoor_objects,
		enable_outdoor_objects=not args.no_load_outdoor_objects,
		outdoor_objects_max_num=args.outdoor_objects_max_num,
		enable_collision=args.enable_collision,
		enable_decompose=args.enable_decompose,
		skip_avatar_animation=args.skip_avatar_animation,
		enable_gt_segmentation=args.enable_gt_segmentation,
		output_dir=args.output_dir,
		enable_third_person_cameras=args.enable_third_person_cameras,
		enable_demo_camera=args.enable_demo_camera,
		no_traffic_manager=args.no_traffic_manager,
		enable_tm_debug=args.enable_tm_debug,
		tm_vehicle_num=args.tm_vehicle_num,
		tm_avatar_num=args.tm_avatar_num,
		save_per_seconds=args.save_per_seconds,
		defer_chat="ella" in args.agent_type or "no_" in args.agent_type,
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
			no_react = False,
			debug = args.debug,
			logging_level = args.logging_level,
			multi_process = args.multi_process,
		)
		llm_kwargs = dict(
			lm_source=args.lm_source,
			lm_id=args.lm_id
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
