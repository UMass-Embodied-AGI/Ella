import os
import sys
import time
import shutil, errno
import argparse
import genesis as gs
import json

current_directory = os.getcwd()
sys.path.insert(0, current_directory)
from vico.env import VicoEnv
from vico.tools.utils import atomic_save, json_converter


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
	parser.add_argument("--config", type=str, default='agents_num_15_with_schedules')
	parser.add_argument("--agent_type", type=str, choices=['ella'], default='ella')
	parser.add_argument("--detect_interval", type=int, default=1)
	parser.add_argument("--region_layer", action='store_true')

	parser.add_argument("--lm_source", type=str, choices=["openai", "azure", "huggingface", "local"]
						, default="azure", help="language model source")
	parser.add_argument("--lm_id", "-lm", type=str, default="gpt-4o", help="language model id")

	args = parser.parse_args()

	args.output_dir = os.path.join(args.output_dir, f"{args.scene}_{args.config}", f"{args.agent_type}")

	if args.overwrite and os.path.exists(args.output_dir):
		print(f"Overwrite the output directory: {args.output_dir}")
		shutil.rmtree(args.output_dir)
	os.makedirs(args.output_dir, exist_ok=True)
	config_path = os.path.join(args.output_dir, 'curr_sim')
	if not os.path.exists(config_path):
		seed_config_path = os.path.join('vico/assets/scenes', args.scene, args.config)
		print(f"Initiate new simulation from config: {seed_config_path}")
		try:
			shutil.copytree(seed_config_path, config_path)
		except OSError as exc:
			if exc.errno in (errno.ENOTDIR, errno.EINVAL):
				shutil.copy(seed_config_path, config_path)
			else:
				raise
	else:
		print(f"Continue simulation from config: {config_path}")

	config = json.load(open(os.path.join(config_path, 'config.json'), 'r'))
	if args.debug:
		args.enable_third_person_cameras = True
		if args.curr_time is not None:
			config['curr_time'] = args.curr_time
			atomic_save(os.path.join(config_path, 'config.json'), json.dumps(config, indent=4, default=json_converter))

	os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)

	from tools.model_manager import global_model_manager
	global_model_manager.init(local=True)

	env = VicoEnv(
		seed=args.seed,
		precision=args.precision,
		logging_level=args.logging_level,
		backend= gs.cpu if args.backend == 'cpu' else gs.gpu,
		head_less=args.head_less,
		resolution=args.resolution,
		challenge=args.challenge,
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
		defer_chat=True,
		debug=args.debug,
		batch_renderer=args.batch_renderer,
	)
	from agents import AgentProcess, get_agent_cls_ella
	agents = []
	for i in range(args.num_agents):
		basic_kwargs = dict(
			name = env.agent_names[i],
			pose = env.config["agent_poses"][i],
			info = env.agent_infos[i],
			sim_path = config_path,
			debug = args.debug,
			logging_level = args.logging_level,
			multi_process = args.multi_process
		)
		agent_cls = get_agent_cls_ella(agent_type=args.agent_type)
		agents.append(AgentProcess(agent_cls, **basic_kwargs))

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
			if i in agent_list_to_update:
				agent.update(obs[i])
		for i, agent in enumerate(agents):
			if i in agent_list_to_update:
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

	for agent in agents:
		agent.close()
	env.close()
