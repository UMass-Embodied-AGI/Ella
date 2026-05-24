import argparse
import datetime
import os
import json
from collections import defaultdict

import numpy as np

import tqdm
from vico.tools.utils import merge_step_files
from vico.tools.constants import ASSETS_PATH


def evaluate_IB(output_path, max_steps=21600):
    config = json.load(open(os.path.join(output_path, "curr_sim", "config.json"), 'r'))
    scene = config["sim_name"].split('_')[0]
    place_meta = json.load(open(os.path.join(ASSETS_PATH, "scenes", scene, "place_metadata.json"), 'r'))
    party_organizer_groups_name = config["party_organizer_groups"]
    party_organizer_groups_place = [config["groups"][group]["place"] for group in party_organizer_groups_name]
    total_conv = defaultdict(int)
    show_up_agents = defaultdict(set)
    party_start_time = datetime.datetime.strptime("October 02, 2024, 14:29:59", "%B %d, %Y, %H:%M:%S")
    party_end_time = datetime.datetime.strptime("October 02, 2024, 15:00:00", "%B %d, %Y, %H:%M:%S")
    all_steps_json = merge_step_files(os.path.join(output_path, "steps"), config["agent_names"], config["step"]) # , overwrite=True)
    steps = min(max_steps, config["step"])
    for agent_name in config["agent_names"]:
        agent_scratch = json.load(open(os.path.join(output_path, "curr_sim", agent_name, "scratch.json"), 'r'))
        group_name = agent_scratch["groups"][0]["name"]
        for step in tqdm.tqdm(range(steps)):
            try:
                step_data = all_steps_json[agent_name][step]
            except Exception as e:
                if step != config["step"]:
                    print(f"At step {step}, Error: {e}")
                continue
            step_time = datetime.datetime.strptime(step_data["curr_time"], "%B %d, %Y, %H:%M:%S")
            if step_time > party_end_time:
                break
            if step_time > party_start_time:
                if step_data["obs"]["current_place"] in party_organizer_groups_place:
                    show_up_agents[step_data["obs"]["current_place"]].add(agent_name)
                else:
                    for group_place in party_organizer_groups_place:
                        if place_meta[group_place]["building"] == "open space":
                            if np.linalg.norm(np.array(place_meta[group_place]["location"][:2]) - np.array(step_data["obs"]["pose"][:2])) < 5:
                                show_up_agents[group_place].add(agent_name)
                                break

            action = step_data.get("action")
            if action and isinstance(action, dict) and action.get("type") == "converse":
                total_conv[group_name] += 1

    all_show_up_agents = set()
    for group_place in party_organizer_groups_place:
        all_show_up_agents.update(show_up_agents[group_place])
    all_show_up_agents = list(all_show_up_agents)
    show_up_rate = len(all_show_up_agents) / len(config["agent_names"]) * 100
    all_conversations = 0
    party_organizer_groups = {}
    for group_name, group_place in zip(party_organizer_groups_name, party_organizer_groups_place):
        all_conversations += total_conv[group_name]
        party_organizer_groups[group_name] = config["groups"][group_name]
        party_organizer_groups[group_name].pop("description")
        party_organizer_groups[group_name].pop("daily_activity")
        party_organizer_groups[group_name]["show_up_agents"] = sorted(list(show_up_agents[group_place]))
    print(f"Agents showed up: {party_organizer_groups}\nTotal conversations: {total_conv}\nshow up rate: {show_up_rate}\nall conversations: {all_conversations}")
    json.dump({"show_up_rate": show_up_rate, "all_conversations": all_conversations, "total_conv": total_conv,
                "party_organizer_groups": party_organizer_groups, "groups": config["groups"]}, open(os.path.join(output_path, "results.json"), 'w'), indent=2)


def evaluate_LQ(output_path, max_steps=10800):
    config = json.load(open(os.path.join(output_path, "curr_sim", "config.json"), 'r'))
    results = config["group_task_info"]
    if "goods_categories" in results:
        results.pop("goods_categories")
    for group_name, group_task_info in results.items():
        total_target_num = 0
        total_achieved_num = 0
        for target_goods_num, achieved_goods_num in zip(group_task_info["target_goods"], group_task_info["achieved_goods"]):
            achieved_goods_num = min(achieved_goods_num, target_goods_num)
            total_achieved_num += achieved_goods_num
            total_target_num += target_goods_num
        group_task_info["Completion_rate"] = total_achieved_num / total_target_num
        total_cost = 0
        for agent_name in config["groups"][group_name]["members"]:
            agent_idx = config["agent_names"].index(agent_name)
            total_cost += 50 - config["agent_infos"][agent_idx]["cash"]
        group_task_info["Total_cost"] = total_cost
        group_task_info["Total_conv"] = 0

    all_steps_json = merge_step_files(os.path.join(output_path, "steps"), config["agent_names"], config["step"])
    steps = min(max_steps, config["step"])
    for agent_name in config["agent_names"]:
        agent_scratch = json.load(open(os.path.join(output_path, "curr_sim", agent_name, "scratch.json"), 'r'))
        group_name = agent_scratch["groups"][0]["name"]
        for step in tqdm.tqdm(range(steps)):
            try:
                step_data = all_steps_json[agent_name][step]
            except Exception as e:
                # if step != config["step"]:
                #     print(f"At step {step}, Error: {e}")
                continue

            action = step_data.get("action")
            if action and isinstance(action, dict) and action.get("type") == "converse":
                results[group_name]["Total_conv"] += 1

    avg_completion_rate = sum([group_task_info["Completion_rate"] for group_task_info in results.values()]) / len(results)
    avg_total_cost = sum([group_task_info["Total_cost"] for group_task_info in results.values()])
    avg_total_conv = sum([group_task_info["Total_conv"] for group_task_info in results.values()])
    results["avg_completion_rate"] = avg_completion_rate * 100
    results["avg_total_cost"] = avg_total_cost
    results["avg_total_conv"] = avg_total_conv
    print(json.dumps(results, indent=2))
    json.dump(results, open(os.path.join(output_path, "results.json"), 'w'), indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="output/CF/IB/newyork_agents_num_15/ella")
    parser.add_argument("--challenge", "-c", type=str, required=True)
    parser.add_argument("--scenes", "-s", type=str, nargs="+", default=["newyork"])
    parser.add_argument("--agents", "-a", type=str, nargs="+", default=["ella"])
    args = parser.parse_args()
    if args.challenge == "IB":
        evaluate_IB(args.data_dir, 21600)
    elif args.challenge == "LQ":
        evaluate_LQ(args.data_dir, 10800)
    elif args.challenge == "TH":
        pass
