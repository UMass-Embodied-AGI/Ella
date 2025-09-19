import argparse
import json
from datetime import datetime
import os
import sys
import shutil

current_directory = os.getcwd()
sys.path.insert(0, current_directory)

from agents.ella import EllaAgent
from tools.model_manager import global_model_manager

# init logger
import logging
logger = logging.getLogger('ella')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh = logging.FileHandler('a.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

parser = argparse.ArgumentParser()
parser.add_argument("--scene", "-s", type=str, default='NY')
parser.add_argument("--num_agents", "-n", type=int, default=15)
parser.add_argument("--config", type=str, default='agents_num_15')
parser.add_argument("--event", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--debug", action='store_true')
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--curr_time", type=str)
args = parser.parse_args()

if not args.output_dir:
	args.output_dir = os.path.join("vico/assets/scenes", args.scene, args.config + "_with_schedules")
	if args.event:
		args.output_dir = os.path.join("vico/assets/scenes", args.scene, "events", args.event, args.config + "_with_schedules")
if args.overwrite and os.path.exists(args.output_dir):
	print(f"Overwrite the output directory: {args.output_dir}")
	shutil.rmtree(args.output_dir)

seed_sim_path = f"vico/assets/scenes/{args.scene}/{args.config}"
if args.event:
	seed_sim_path = f"vico/assets/scenes/{args.scene}/events/{args.event}/{args.config}"
seed_config_path = os.path.join(seed_sim_path, "config.json")


config_path = os.path.join(args.output_dir, "config.json")
if not os.path.exists(args.output_dir):
	shutil.copytree(seed_sim_path, args.output_dir)

config = json.load(open(config_path, "r"))

global_model_manager.init(local=True)
for i in range(args.num_agents):
	agent = EllaAgent(name=config["agent_names"][i], pose=config["agent_poses"][i], info=config["agent_infos"][i],
					  sim_path=args.output_dir, no_react=True, debug=True, logger=logger, lm_source='azure', lm_id='gpt-4o', detect_interval=-1)
	hourly_schedule_path = os.path.join(agent.storage_path, "hourly_schedule.json")
	if os.path.exists(hourly_schedule_path):
		print(f"Skip generating schedule for {agent.name}")
		schedule = json.load(open(hourly_schedule_path, "r"))
	else:
		# if args.curr_time is not None:
		# 	agent.set_curr_time(datetime.strptime(args.curr_time, "%H:%M:%S"))
		schedule = agent.generate_hourly_schedule()
		with open(hourly_schedule_path, "w") as f:
			json.dump(schedule, f, indent=4)
	agent.hourly_schedule = schedule
	agent.save_scratch()
print(f"Generated schedules for {args.num_agents} agents in {args.scene}.")
os._exit(0)
