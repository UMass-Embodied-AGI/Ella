import os
import sys
import time
import traceback
import json
from typing import Optional
import numpy as np
import random
from datetime import datetime, timedelta
from PIL import Image
from dataclasses import dataclass
from .memory import SemanticMemory, EpisodicMemory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vico.agents import Agent
from vico.tools.utils import *

@dataclass
class Chat:
	time: datetime
	subject: str
	pos: list
	content: str

	def to_dict(self):
		return {
			"time": self.time.strftime("%B %d, %Y, %H:%M:%S"),
			"subject": self.subject,
			"pos": self.pos,
			"content": self.content
		}

class EllaAgent(Agent):
	def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
				 lm_source='azure', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0, detect_interval=1, region_layer=False, enable_indoor_activities=False):
		super().__init__(name, pose, info, sim_path, no_react, debug, logger)
		from tools.model_manager import global_model_manager
		self.lm_source = lm_source
		self.generator = global_model_manager.get_generator(lm_source, lm_id, max_tokens, temperature, top_p, logger)
		# self.embedding_generator = Generator(lm_source, 'embedding', max_tokens, temperature, top_p, logger)
		self.react_freq = self.scratch["react_freq"] if "react_freq" in self.scratch else 1e8
		self.motion_schedule = None
		self.enable_indoor_activities = enable_indoor_activities
		if self.debug:
			self.react_freq = 300 # 5 min for debug
		if self.no_react:
			self.react_freq = 1e8
		self.chat_time_limit = 15 # 15 seconds
		self.hourly_schedule = self.scratch["hourly_schedule"]
		self.curr_schedule_idx = self.get_curr_schedule_idx()
		self.motion_place = None
		self.curr_motion_schedule_idx = 0
		self.curr_motion_schedule_idx2 = 0
		if "commute_plan" in self.scratch:
			self.commute_plan = self.scratch["commute_plan"]
			self.commute_plan_idx = self.scratch["commute_plan_idx"]
		else:
			self.commute_plan = None
			self.commute_plan_idx = 0
		if "last_enter_bus_time" in self.scratch and self.scratch["last_enter_bus_time"] is not None:
			self.last_enter_bus_time = datetime.strptime(self.scratch["last_enter_bus_time"], "%B %d, %Y, %H:%M:%S")
		else:
			self.last_enter_bus_time = None

		self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), detect_interval=detect_interval, region_layer=region_layer, debug=self.debug, logger=self.logger) #todo: log the num_frames
		self.e_mem = EpisodicMemory(os.path.join(self.storage_path, "episodic_memory"), lm_source, debug=self.debug, logger=self.logger)
		lastevent = self.e_mem.get_memory(-1)
		if lastevent is not None:
			self.last_react_time = lastevent.event_time
		else:
			self.last_react_time = None

		chatting_buffer = self.scratch["chatting_buffer"] if "chatting_buffer" in self.scratch else []
		self.chatting_buffer: list[Chat] = [Chat(datetime.strptime(chat["time"], "%B %d, %Y, %H:%M:%S"), chat["subject"], chat["pos"], chat["content"]) for chat in chatting_buffer]
		self.react_mode = None
		self.last_action = None
		self.last_adjust_schedule_time = self.e_mem.retrieve_memory_by_keyword("schedule")
		self.last_adjust_schedule_time = datetime.strptime(self.last_adjust_schedule_time[-1]["event_time"], "%B %d, %Y, %H:%M:%S") if self.last_adjust_schedule_time is not None else None

	def reset(self, name, pose):
		super().reset(name, pose)
		self.hourly_schedule = self.scratch["hourly_schedule"]
		self.curr_schedule_idx = self.get_curr_schedule_idx()
		self.commute_plan = None
		self.commute_plan_idx = 0
		
		self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), debug=self.debug, logger=self.logger)
		self.e_mem = EpisodicMemory(os.path.join(self.storage_path, "episodic_memory"), self.lm_source, debug=self.debug, logger=self.logger)

	def set_curr_time(self, curr_time: datetime):
		self.curr_time = curr_time
		self.curr_schedule_idx = self.get_curr_schedule_idx()

	def update_chatting_with(self):
		for chat in self.chatting_buffer:
			if chat.subject == "someone out of sight":
				subject = self.s_mem.get_name_from_position(chat.pos)
				if subject is not None:
					chat.subject = subject
					self.logger.info(f"Update chatting with {subject} at {chat.pos}.")

	def get_chatting_with(self) -> list:
		return list(set([chat.subject for chat in self.chatting_buffer]))

	def _process_obs(self, obs):
		# if new day, generate new hourly schedule
		if obs['new_day'] or self.hourly_schedule == []:
			self.hourly_schedule = self.generate_hourly_schedule()

			self.curr_schedule_idx = 0
		start = time.time()
		if obs['action_status'] == "FAIL":
			self.logger.info(f"{self.name} failed to execute last action {self.last_action}.")
			self.e_mem.update_memory_last_action()
			if self.last_action["type"] == "converse":
				if len(self.chatting_buffer) > 0 and self.chatting_buffer[-1].subject == self.name:
					self.chatting_buffer.pop()

		self.last_action = None
		num_new_objects = self.s_mem.update(obs)
		self.logger.debug(f"Process obs 2: {start}, {time.time()}")

		# react to new objects
		if not self.no_react and num_new_objects > 0:
			new_objects = self.s_mem.object_builder.get_new_objects()
			curr_objects = self.s_mem.object_builder.get_curr_objects()
			kws = [object.name for object in curr_objects]
			img_path = os.path.join(self.storage_path, 'episodic_memory',
									f'img_{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}.png')
			Image.fromarray(obs['rgb']).save(img_path)
			if "gt_seg_entity_idx_to_info" in obs:
				desc = f"I see {', '.join([object.name for object in curr_objects])}."
			else:
				desc = self.generate_captioning(
					f"Here's an image including {', '.join([object.name for object in new_objects])}. Describe what you see in one sentence. Start with 'I see'.",
					img=img_path)
				desc += f" Entities detected: {', '.join([object.name for object in curr_objects])}."
			self.e_mem.add_memory("observation", self.curr_time, self.pose[:3], obs['current_place'], kws, img_path,
								  desc, None)
			self.last_react_time = self.curr_time
		start = time.time()
		if self.chatting_buffer:
			for event in obs['events']:
				if event["type"] == "speech":
					if event["position"][:2] == self.pose[:2]: # ignore self speech
						continue
					subject = self.s_mem.get_name_from_position(event["position"])
					if subject is None:
						subject = "someone out of sight"
					self.chatting_buffer.append(Chat(self.curr_time, subject, event["position"], event["content"]))
					self.logger.info(f"{self.name} hears {subject} at {event['position']} says: {event['content']}")
			self.update_chatting_with()
			return
		self.logger.debug(f"Process obs 1: {start}, {time.time()}")
		start = time.time()
		# react[also save episodic mem] every react_freq seconds or new objects appear
		if len(obs['events']) > 0:
			for event in obs['events']:
				if event["type"] == "speech":
					if event["position"][:2] == self.pose[:2]:
						continue
					subject = self.s_mem.get_name_from_position(event["position"])
					if subject is None:
						subject = "someone out of sight"
					event["content"] = f"I heard {subject} at {event['position']} says: {event['content']}"
					kws = [subject, event['type']]
				else:
					kws = [event["type"]]

				if obs['rgb'] is not None:
					img_path = os.path.join(self.storage_path, 'episodic_memory', f'img_{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}.png')
					Image.fromarray(obs['rgb']).save(img_path)
				else:
					img_path = None

				self.e_mem.add_memory(event["type"], self.curr_time, event["position"], obs['current_place'], kws, img_path, event["content"], None)
			# need to react to the events
			self.last_react_time = self.curr_time

		if not self.no_react and (self.last_react_time is None or (self.last_react_time != self.curr_time and (self.curr_time - self.last_react_time).total_seconds() > self.react_freq)):
			if obs['rgb'] is not None:

				# todo: get the keywords
				donot_add = False

				img_path = os.path.join(self.storage_path, 'episodic_memory',
										f'img_{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}.png')
				Image.fromarray(obs['rgb']).save(img_path)
				if "gt_seg_entity_idx_to_info" in obs:
					desc = f"I see {', '.join([object.name for object in self.s_mem.object_builder.get_curr_objects()])}."
					kws = [object.name for object in self.s_mem.object_builder.get_curr_objects()]
					if not kws:
						donot_add = True
				else:
					desc = self.generate_captioning(f"Describe what you see in one sentence. Start with 'I see'.", img=img_path)
					kws = []
				if not donot_add:
					self.e_mem.add_memory("observation", self.curr_time, self.pose[:3], obs['current_place'], [], img_path, desc, None)
					self.last_react_time = self.curr_time

		self.logger.debug(f"Process obs 3: {start}, {time.time()}")

	def _act(self, obs):
		start = time.time()
		self.react_mode = None
		if self.chatting_buffer:
			action = self.conversation(None)
			self.logger.debug(f"Generate conversation action time: {time.time() - start}")
			if action is not None:
				self.last_action = action
				return action
		if self.curr_schedule_idx >= len(self.hourly_schedule):
			self.logger.info(f"{self.name} has finished the daily plan.")
			return None
		while obs['curr_time'].time() >= datetime.strptime(self.hourly_schedule[self.curr_schedule_idx]["end_time"], "%H:%M:%S").time():
			self.curr_schedule_idx += 1
			if self.curr_schedule_idx >= len(self.hourly_schedule):
				self.logger.info(f"{self.name} has finished the daily plan.")
				return None
			self.commute_plan = None

		action = None

		# react to the curr_events related retrieved events
		start = time.time()
		if not self.no_react and self.last_react_time == self.curr_time:
			curr_events = self.e_mem.retrieve_latest_memory()
			retrieved_events = self.e_mem.retrieve("important things to react to", obs["rgb"], self.curr_time, self.pose[:3], 3)
			self.react_mode, react_target, react_reason = self.generate_react_mode(curr_events, retrieved_events)
			if self.react_mode == "continue doing current activity":
				pass
			elif self.react_mode == "engage in a conversation":
				self.chatting_buffer = []
				action = self.conversation(react_target)
			elif self.react_mode == "adjust the schedule":
				action = self.adjust_schedule(react_reason)
			elif self.react_mode == "buy a good":
				if react_target is not None:
					if self.held_objects[0] is None:
						action = {
							'type': 'fake_pick',
							'arg1': 0,
							'arg2': react_target
						}
						self.e_mem.add_memory("action", self.curr_time, self.pose[:3], self.current_place, ['pick', react_target], None, f"Pick up {react_target} at {self.current_place}.", None)
					elif self.held_objects[1] is None:
						action = {
							'type': 'fake_pick',
							'arg1': 1,
							'arg2': react_target
						}
						self.e_mem.add_memory("action", self.curr_time, self.pose[:3], self.current_place, ['pick', react_target], None, f"Pick up {react_target} at {self.current_place}.", None)
					else:
						self.logger.warning(f"I already held two objects {self.held_objects}, cannot pick up {react_target}.")
			else:
				pass
		elif self.hourly_schedule[self.curr_schedule_idx]["place"] is not None and self.hourly_schedule[self.curr_schedule_idx]["building"] != "open space" and self.hourly_schedule[self.curr_schedule_idx]["place"] != self.current_place:
			action = self.adjust_schedule()
		self.logger.debug(f"React time: {start}, {time.time()}")

		if action is not None:
			self.last_action = action
			return action
		# follow the schedule
		if self.hourly_schedule[self.curr_schedule_idx]["type"] == "commute":
			if self.commute_plan is None:
				self.commute_plan = self.generate_commute_plan()
				self.commute_plan_idx = 0
			action =  self.commute()
		else:
			if self.current_place == self.scratch["groups"][0]["place"]:
				if self.held_objects[0] is not None:
					action = {
						'type': 'fake_put',
						'arg1': 0
					}
					self.logger.info(f"{self.name} is at the group place, put down the object in hand 0.")
				elif self.held_objects[1] is not None:
					action = {
						'type': 'fake_put',
						'arg1': 1
					}
					self.logger.info(f"{self.name} is at the group place, put down the object in hand 1.")
			# print (f"acting! {self.current_place}, {self.motion_place}")
			if action is None and self.enable_indoor_activities:
				if self.current_place != self.motion_place:
					self.motion_place = self.current_place
					self.curr_motion_schedule_idx = 0
					self.motion_schedule = None
				if not self.s_mem.explored.get(self.current_place, None):
					action = self.explore()
				else:
					self.logger.info(f"{self.name}'s schedule now: {self.hourly_schedule[self.curr_schedule_idx]}.")
					action = self.get_motion_from_schedule(obs['curr_time'].time())
					if self.curr_motion_schedule_idx < len(self.motion_schedule):
						self.logger.info(f"{self.name} is at {self.current_place} for {self.motion_schedule[self.curr_motion_schedule_idx]}.")
					self.logger.info(f"a new action: {action}.")

		if action is None:
			action = {"type": "wait", "arg1": None}
		self.last_action = action
		return action

	def explore(self):
		motion_list = [
			{
				"type": "turn_left",
				"arg1": 90
			},
			{
				"type": "turn_left",
				"arg1": 90
			},
			{
				"type": "turn_left",
				"arg1": 90
			},
		]
		action = motion_list[self.curr_motion_schedule_idx]
		self.curr_motion_schedule_idx += 1
		if self.curr_motion_schedule_idx >= len(motion_list):
			self.logger.info(f"{self.name} has finished the exploration.")
			self.s_mem.explored[self.current_place] = True
		return action

	def get_motion_from_schedule(self, curr_time):
		if self.motion_schedule is None or self.current_place != self.motion_place:
			self.motion_schedule = self.generate_motion_schedule()
			self.curr_motion_schedule_idx = 0
			self.curr_motion_schedule_idx2 = 0
		if self.curr_motion_schedule_idx >= len(self.motion_schedule):
			self.logger.info(f"{self.name} has finished the motion schedule.")
			return None
		if curr_time < datetime.strptime(self.motion_schedule[self.curr_motion_schedule_idx]["time"], "%H:%M:%S").time():
			self.logger.debug(f"{self.name} is waiting for the next motion schedule.")
			return None
		
		action = self.motion_schedule[self.curr_motion_schedule_idx]['motion_list'][self.curr_motion_schedule_idx2]

		def navigate_to(goal_pos, goal_bbox, threshold=0.4):
			goal_pos = np.array(goal_pos)
			if goal_bbox is None:
				goal_bbox = np.array([goal_pos - threshold / 2, goal_pos + threshold / 2])
				
			x1, y1, z1 = goal_bbox[0]
			x2, y2, z2 = goal_bbox[1]

			points = [[x1, y1, z1],[x1, y2, z1],[x2, y1, z1],[x2, y2, z1],
						[x1, y1, z2],[x1, y2, z2],[x2, y1, z2],[x2, y2, z2]]
			goal_bbox = np.array(points) 
			goal_bbox = bbox_corners_to_center_repr(goal_bbox)

			navigate_action = self.navigate(self.s_mem.get_sg(self.current_place), goal_pos, goal_bbox)
			dist = np.linalg.norm(np.array(self.pose[:2]) - np.array(goal_pos[:2]))
			self.logger.info(f"Navigate to {goal_pos} with distance {dist}.")
			near_goal = dist <= threshold
			stucked = (navigate_action['type'] == "move_forward" and navigate_action['arg1'] < 0.1)
			if near_goal or stucked:
				if near_goal:
					self.logger.info(f"{self.name} arrived at the goal {goal_pos} with current location {self.pose}.")
				if stucked:
					self.logger.info(f"{self.name} is stucked navigating the goal {goal_pos} with current location {self.pose}.")
				return None
			return navigate_action
		
		def step_to_next_motion():
			self.curr_motion_schedule_idx2 += 1
			if self.curr_motion_schedule_idx2 >= len(self.motion_schedule[self.curr_motion_schedule_idx]['motion_list']):
				self.curr_motion_schedule_idx += 1
				self.curr_motion_schedule_idx2 = 0

		def random_point_in_center(x_min, x_max, y_min, y_max):
			x_center_min = x_min + 0.1 * (x_max - x_min)
			x_center_max = x_max - 0.1 * (x_max - x_min)
			y_center_min = y_min + 0.1 * (y_max - y_min)
			y_center_max = y_max - 0.1 * (y_max - y_min)

			x = random.uniform(x_center_min, x_center_max)
			y = random.uniform(y_center_min, y_center_max)
			return [x, y]

		objects = self.s_mem.scene_graph_dict[self.s_mem.current_place].objects
		try:
			if action['type'] == "navigate_to":
				if isinstance(action['arg1'], str):
					action['raw'] = action['arg1']
					action['debug_box'] = objects[action['arg1']].get_bound()
					action['arg1'] = (objects[action['arg1']].get_position(),objects[action['arg1']].get_bound())
				goal_pos, goal_bbox = action['arg1']
				navigate_action = navigate_to(goal_pos, goal_bbox)
				if navigate_action is None:
					step_to_next_motion()
					return self.get_motion_from_schedule(curr_time)
				navigate_action = {**action, **navigate_action}
				return navigate_action
			
			if action['type'] in ['put', 'pick']:
				if isinstance(action['arg2'], str):
					action['raw'] = action['arg2']
					action['need_navigate'] = True
					if action['type'] == 'pick':
						self.objects_on_hands[action['arg1']] = action['arg2']
						action['raw'] = objects[action['arg2']].name
						action['debug_box'] = objects[action['arg2']].get_bound()
						action['arg2'] = objects[action['arg2']].get_position()
					else:
						target_pos = objects[action['arg2']].get_bound()
						action['debug_box'] = objects[action['arg2']].get_bound()
						x_min, x_max, y_min, y_max = target_pos[0][0], target_pos[1][0], target_pos[0][1], target_pos[1][1]
						action['arg2'] = np.array(random_point_in_center(x_min, x_max, y_min, y_max) + [target_pos[1][2]])
						action['debug_sphere'] = action['arg2']
				if action['need_navigate']:
					navigate_action = navigate_to(action['arg2'], None)
					if navigate_action is not None:
						return {**action, **navigate_action}
					action['need_navigate'] = False
			elif action['type'] in ['sit']:
				if isinstance(action['arg1'], str):
					action['raw'] = action['arg1']
					action['need_navigate'] = True
					action['debug_box'] = objects[action['arg1']].get_bound()
					action['arg1'] = (objects[action['arg1']].get_position(), objects[action['arg1']].get_bound())
				if action['need_navigate']:
					goal_pos, goal_bbox = action['arg1']
					navigate_action = navigate_to(goal_pos, goal_bbox, threshold=0.1)
					if navigate_action is not None:
						return {**action, **navigate_action}
					action['need_navigate'] = False
			elif action['type'] in ['look_at']:
				if isinstance(action['arg1'], str):
					action['raw'] = action['arg1']
					action['need_navigate'] = True
					action['debug_box'] = objects[action['arg1']].get_bound()
					action['arg1'] = objects[action['arg1']].get_position()
			else:
				action['raw'] = "None"
			step_to_next_motion()
			return action
		except Exception as e:
			self.logger.error(f"Error in get_motion_from_schedule: {e}")

			step_to_next_motion()

	def chat(self, content):
		target = content
		target_knowledge = self.s_mem.get_knowledge(target)
		if target_knowledge is None:
			self.logger.warning(f"No knowledge found for {target}.")
		target_experience = self.e_mem.retrieve_memory_by_keyword(target)
		if target_experience is None:
			self.logger.warning(f"No experience found for {target}.")
		utterance = self.generate_utterance(target, target_knowledge, target_experience)
		if utterance is None:
			self.end_conversation()
			return None
		self.chatting_buffer.append(Chat(self.curr_time + timedelta(seconds=1), self.name, self.pose[:3], utterance))
		return utterance

	def commute(self):
		# if current time > next start time:
		if self.curr_time.time() > datetime.strptime(self.hourly_schedule[self.curr_schedule_idx + 1]["start_time"], "%H:%M:%S").time():
			self.logger.warning(f"{self.name} is late for the next activity {self.hourly_schedule[self.curr_schedule_idx + 1]['activity']}.")
			self.commute_plan = [{
				"goal_place": self.hourly_schedule[self.curr_schedule_idx + 1]["place"],
				"transit_type": "walk"
			}]
			self.commute_plan_idx = 0
			return None
		if self.commute_plan_idx >= len(self.commute_plan):
			self.logger.debug(f"{self.name} finished the commute plan.")
			self.curr_schedule_idx += 1
			self.commute_plan = None
			self.commute_plan_idx = 0
			return None
		if self.commute_plan[self.commute_plan_idx]["transit_type"] == "bus":
			if self.current_vehicle != "bus":
				if "bus" in self.obs['accessible_places']:
					self.last_enter_bus_time = self.curr_time
					return {
						'type': 'enter_bus',
						'arg1': None
					}
				else:
					self.logger.debug(f"Agent {self.name} at {self.pose} is waiting for the bus.")
					return {
						'type': 'wait',
						'arg1': None
					}
			if (self.commute_plan[self.commute_plan_idx]["goal_place"] in self.obs['accessible_places'] or
					int((self.curr_time - self.last_enter_bus_time).total_seconds() / 60) > 10 and len(self.obs['accessible_places']) > 0):
				self.commute_plan_idx += 1
				self.last_enter_bus_time = None
				return {
					'type': 'exit_bus',
					'arg1': None
				}
			else:
				self.logger.debug(f"Agent {self.name} at {self.pose} is riding the bus.")
				return None
		elif self.commute_plan[self.commute_plan_idx]["transit_type"] == "bike":
			if self.current_vehicle != "bicycle":
				if "bicycle" in self.obs['accessible_places']:
					return {
						'type': 'enter_bike',
						'arg1': None
					}
				else:
					self.logger.warning(f"Agent {self.name} at {self.pose} didn't find any bike! change commute type to walk")
					self.commute_plan[self.commute_plan_idx]["transit_type"] = "walk"
			else:
				if self.commute_plan[self.commute_plan_idx]["goal_place"] in self.obs['accessible_places']:
					self.commute_plan_idx += 1
					self.logger.debug(f"Agent {self.name} at {self.pose} arrives at {self.commute_plan[self.commute_plan_idx]['goal_place']} and is returning the bike.")
					return {
						'type': 'exit_bike',
						'arg1': None
					}
				else:
					self.logger.debug(f"Agent {self.name} at {self.pose} is riding the bike.")
		elif self.commute_plan[self.commute_plan_idx]["transit_type"] == "walk":
			if self.current_vehicle == "bicycle":
				self.logger.warning(f"Agent {self.name} at {self.pose} need to return the bike to the closest station first.")
				self.commute_plan.insert(self.commute_plan_idx, {
					"goal_place": self.get_closest_bike_station(self.pose[:2]),
					"transit_type": "bike"
				})
				return self.commute()
			if self.current_vehicle == "bus":
				self.logger.warning(f"Agent {self.name} at {self.pose} need to exit the bus first.")
				self.commute_plan.insert(self.commute_plan_idx, {
					"goal_place": self.get_closest_bus_stop(self.pose[:2]),
					"transit_type": "bus"
				})
				return self.commute()
			pass
		else:
			self.logger.error(f"Unknown commute type {self.commute_plan[self.commute_plan_idx]['type']}.")
			return None
		goal_place = self.commute_plan[self.commute_plan_idx]["goal_place"]

		goal_place_dict = self.s_mem.get_knowledge(goal_place)
		if goal_place_dict is None:
			self.logger.error(f"No knowledge found for {goal_place}.")
			return None
		goal_pos = np.array([goal_place_dict["location"][0], goal_place_dict["location"][1]])
		goal_bbox = goal_place_dict["bounding_box"]
		self.logger.debug(f"Goal place: {goal_place}, goal pos: {goal_pos}, goal bbox: {goal_bbox}")

		self.last_action = {'type': 'wait', 'arg1': None}
		# already at the correct place
		if goal_place == self.obs['current_place']:
			self.commute_plan_idx += 1
			self.logger.debug(
				f"{self.name} arrived at {goal_place} for {self.hourly_schedule[self.curr_schedule_idx + 1]['activity']}.")
			return self.last_action
		# can enter the correct place
		if goal_place in self.obs['accessible_places']:
			self.commute_plan_idx += 1
			self.logger.debug(
				f"{self.name} finished navigation to {goal_place} at {goal_pos}")
			self.last_action = {
				'type': 'enter',
				'arg1': goal_place
			}
			return self.last_action
		# at wrong place, need to enter open space
		if self.obs['current_place'] is not None:
			self.logger.debug(
				f"{self.name} at {self.obs['current_place']} is entering open space to move to {goal_place} at {goal_pos}.")
			self.last_action = {
				'type': 'enter',
				'arg1': 'open space'
			}
			return self.last_action

		# at open space, need to move to the correct place
		cur_trans = np.array(self.pose[:2])
		if is_near_goal(cur_trans[0], cur_trans[1], goal_bbox, goal_pos):
			self.logger.warning(
				f"{self.name} at {self.pose} is near the goal {goal_pos}, but not at the goal {goal_place}.")
			return self.last_action
		self.logger.debug(
			f"{self.name} at {tuple(int(p) for p in self.pose)} is moving to {goal_place} at {tuple(int(g) for g in goal_pos)}.")
		start = time.time()
		self.last_action = self.navigate(self.s_mem.get_sg(self.current_place), goal_pos, goal_bbox)
		self.logger.debug(f"Navigate time: {start}, {time.time()}")
		return self.last_action

	def end_conversation(self):
		self.logger.info(f"{self.name} ends the conversation with {self.get_chatting_with()}.")
		summarization = self.generate_conversation_summarization()
		self.e_mem.add_memory("conversation", self.curr_time, self.pose[:3], self.current_place, self.get_chatting_with(), None, summarization, None)
		if len(self.chatting_buffer) > 1:
			new_knowledge = self.generate_new_knowledge()
			if new_knowledge is not None:
				self.s_mem.update_with_new_knowledge(new_knowledge)
		self.chatting_buffer = []

	def conversation(self, target: Optional[str]):
		target_position = None
		if not self.chatting_buffer: # set up the conversation
			curr_events = self.e_mem.retrieve_latest_memory()
			speech_event =  None
			for event in curr_events:
				if event["event_type"] == "speech":
					speech_event = event
					break
			if speech_event:  # response to a conversation
				target = speech_event["event_keywords"][0]
				target_position = speech_event["event_position"]
				self.chatting_buffer.append(
					Chat(self.curr_time, target, target_position, speech_event["event_description"].split("] says: ")[1]))
			else:  # initiate a conversation
				if target is None:
					self.logger.warning(f"target is None when initializing conversation. Fall back to no react.")
					return None
				target_position = self.s_mem.get_position_from_name(target)
				if target_position is None:
					self.logger.warning(f"No position found for {target} when initializing conversation. Fall back to no react.")
					return None
				target_distance = np.linalg.norm(np.array(target_position) - np.array(self.pose[:3]))
				if target_distance > 10:
					self.logger.warning(f"Attempted to talk to {target} {target_distance}m away, fall back to no react.")
					return None
		else:
			# select the last person who talked to me as the target
			for chat in self.chatting_buffer:
				if chat.subject != self.name:
					target = chat.subject
					target_position = chat.pos
		if target == "someone out of sight":
			self.logger.debug(
				f"{self.name} is engaging in a conversation with unknown sound source at {target_position}, turn to it first.")
			return self.turn_to_pos(target_position)
		if self.chatting_buffer and self.chatting_buffer[-1].subject == self.name:
			if (self.curr_time - self.chatting_buffer[-1].time).total_seconds() > 2:
				self.logger.info(f"{self.chatting_buffer[-1].subject} is not responding for more than 2 seconds. Stop chatting.")
				self.end_conversation()
				return None
			return {
				'type': 'wait',
				'arg1': None
			}
		if len(self.chatting_buffer) > self.chat_time_limit:
			self.logger.info(f"Chatting with {self.get_chatting_with()} for more than {self.chat_time_limit} seconds. Stop chatting.")
			self.end_conversation()
			return None
		if target_position is None:
			self.logger.warning(f"No position found for {target}.")
			return None
		else:
			converse_range = np.linalg.norm(np.array(target_position) - np.array(self.pose[:3]))
			if converse_range > 10:
				self.logger.warning(f"Attempted to talk to {target} {converse_range}m away.")
				return None

		return {
			'type': 'converse',
			'arg1': target,
			'arg2': converse_range
		}

	def adjust_schedule(self, react_reason=""):
		current_schedule_place = self.hourly_schedule[self.curr_schedule_idx]["place"]
		remaining_time = (datetime.combine(self.curr_time.date(), datetime.strptime(self.hourly_schedule[self.curr_schedule_idx]["end_time"], "%H:%M:%S").time()) - self.curr_time).total_seconds() / 60
		if remaining_time <= 1:
			self.logger.warning(f"{self.name} is running out of time for the current activity {self.hourly_schedule[self.curr_schedule_idx]['activity']}. Donot adjust schedule.")
			return None
		if len(self.hourly_schedule) > 30:
			self.logger.warning(f"{self.name} has too many activities in the schedule. Skip adjusting the schedule.")
			return None
		if current_schedule_place is not None and current_schedule_place != self.current_place:
			self.logger.warning(f"{self.name} is not at the scheduled place {current_schedule_place}, adjusting the schedule to commute.")
			new_schedule_item ={
				"type": "commute",
				"activity": "Commute to " + current_schedule_place,
				"place": None,
				"building": None,
				"start_time": self.curr_time.strftime("%H:%M:%S"),
				"end_time": (self.curr_time + timedelta(minutes=min(15, remaining_time - 1))).strftime("%H:%M:%S"),
				"start_place": self.current_place,
				"end_place": current_schedule_place
			}
			self.hourly_schedule.insert(self.curr_schedule_idx, new_schedule_item)
			self.hourly_schedule[self.curr_schedule_idx + 1]["start_time"] = new_schedule_item["end_time"]
			self.commute_plan = None
		else:
			if self.last_adjust_schedule_time is not None and (self.curr_time - self.last_adjust_schedule_time).total_seconds() < 60 * 15:
				self.logger.warning(f"Attempted to adjust the schedule too frequently. Skip this time.")
				return None
			retrieved_events = self.e_mem.retrieve("Things I should consider for adjusting my schedule", None, self.curr_time, self.pose[:3], 5)
			adjusted_schedule = self.generate_adjusted_schedule(react_reason, retrieved_events)
			if adjusted_schedule is not None:
				self.hourly_schedule = adjusted_schedule
				self.curr_schedule_idx = self.get_curr_schedule_idx()
				self.logger.info(f"{self.name} adjusted the schedule to {self.hourly_schedule[self.curr_schedule_idx:]}.")
				self.commute_plan = None
				self.e_mem.add_memory("thoughts", self.curr_time, self.pose[:3], self.current_place,
									  ["schedule"], None, f"Adjusted the schedule due to {react_reason}.", None)
				self.last_adjust_schedule_time = self.curr_time
			else:
				self.logger.warning(f"Failed to adjust the schedule. Still use the original schedule.")
		return None

	def generate_captioning(self, prompt, img):
		if self.no_react:
			return "Do not revoke the llm in no react mode."
		response = self.generator.generate(prompt, img=img, json_mode=False)
		return response

	def generate_new_knowledge(self) -> dict[str, dict]:
		prompt = open('agents/prompts/ella/prompt_extract_knowledge_from_conversation.txt', 'r').read()
		prompt = prompt.replace("$Target_name$", ', '.join(self.get_chatting_with()))
		prompt = prompt.replace("$Conversation_history$", '\n'.join([f"{chat.subject}: {chat.content}" for chat in self.chatting_buffer]))
		prompt = prompt.replace("$Knowledge_items$", self.knowledge_name_to_items([self.s_mem.agents[0], self.s_mem.places[0]]))
		self.logger.debug(f"Extract new knowledge from conversation prompt: {prompt}")
		response = self.generator.generate(prompt, img=None, json_mode=False)
		try:
			new_knowledge_items = self.parse_json(prompt, response)
			new_knowledge_items_dict = {item["name"]: item for item in new_knowledge_items}
			# check not empty
			assert len(new_knowledge_items_dict) > 0, f"Zero Knowledge Extracted."
			# assert isinstance(new_knowledge_items, list), f"Invalid new knowledge items: {new_knowledge_items}."
			# for item in new_knowledge_items:
			# 	assert isinstance(item, dict), f"Invalid new knowledge item: {item}."
			# 	assert "name" in item, f"Invalid new knowledge item: {item}."
		except Exception as e:
			self.logger.error(f"Error extracting new knowledge: {e}")
			new_knowledge_items_dict = None

		self.logger.debug(f"new knowledge: {response}")
		return new_knowledge_items_dict

	def generate_conversation_summarization(self):
		prompt = open('agents/prompts/ella/prompt_conversation_summarize.txt', 'r').read()
		prompt = prompt.replace("$Target_name$", ', '.join(self.get_chatting_with()))
		prompt = prompt.replace("$Conversation_history$", '\n'.join([f"{chat.subject}: {chat.content}" for chat in self.chatting_buffer]))
		self.logger.debug(f"Summarization prompt: {prompt}")
		response = self.generator.generate(prompt, img=None, json_mode=False)
		summarization = response
		self.logger.debug(f"Summarization: {response}")
		# try:
		# 	summarization = self.parse_json(response)["summarization"]
		# 	self.logger.info(f"Generated summarization: {summarization}")
		# except Exception as e:
		# 	self.logger.error(f"Error generating summarization: {e}. The response was {response}. Use default summarization.")
		# 	summarization = None
		return summarization

	def generate_utterance(self, target_name, target_knowledge, target_experience):
		prompt = open('agents/prompts/ella/prompt_utterance.txt', 'r').read()
		prompt = prompt.replace("$Character$", self.get_character_description())

		prompt = prompt.replace("$Time$", self.curr_time.strftime("%H:%M:%S"))
		prompt = prompt.replace("$Place$", self.current_place if self.current_place is not None else "open space")
		prompt = prompt.replace("$Target_name$", target_name)
		prompt = prompt.replace("$Target_knowledge$", self.describe_knowledge(target_knowledge))
		prompt = prompt.replace("$Target_experience$", self.describe_events(target_experience))
		conversation_history_desp = '\n'.join([f"{chat.subject}: {chat.content}" for chat in self.chatting_buffer[-4:]])
		if conversation_history_desp == "":
			conversation_history_desp = "No conversation history yet."
			last_chat = f"things to chat about with {target_name}"
		else:
			last_chat = conversation_history_desp.split('\n')[-1]
		prompt = prompt.replace("$Conversation_history$", conversation_history_desp)
		retrieved_events = self.e_mem.retrieve(last_chat, None, self.curr_time, self.pose[:3], 3)
		prompt = prompt.replace("$Context$", self.describe_events(retrieved_events))
		self.logger.debug(f"Utterance prompt: {prompt}")
		response = self.generator.generate(prompt, img=None, json_mode=False)
		try:
			utterance_dict = self.parse_json(prompt, response)
			utterance = utterance_dict["utterance"]
			self.logger.debug(f"Generated utterance: {utterance}\nReason: {utterance_dict['reason']}")
		except Exception as e:
			self.logger.error(f"Error generating utterance: {e}. The response was {response}. Use default utterance.")
			utterance = None
		return utterance

	def generate_react_mode(self, curr_events, retrieved_events):
		prompt = open('agents/prompts/ella/prompt_react.txt', 'r').read()

		prompt = prompt.replace("$Character$", self.get_character_description())
		prompt = prompt.replace("$Schedule$", json.dumps(self.hourly_schedule[self.curr_schedule_idx: min(self.curr_schedule_idx + 3, len(self.hourly_schedule))], indent=2))
		prompt = prompt.replace("$Time$", self.curr_time.strftime("%H:%M:%S"))
		prompt = prompt.replace("$Place$", "None" if self.current_place is None else self.current_place)
		prompt = prompt.replace("$Experiences$", self.describe_events(retrieved_events))
		prompt = prompt.replace("$Context$", self.describe_events(curr_events))
		self.logger.debug(f"React prompt: {prompt}")
		response = self.generator.generate(prompt, img=None, json_mode=False)
		try:
			react_mode_dict = self.parse_json(prompt, response)
			react_mode = react_mode_dict["option"]
			react_target = react_mode_dict["target"]
			react_reason = react_mode_dict["reason"]
			self.logger.debug(f"React mode reason: {react_reason}")
			if react_mode not in ["continue doing current activity", "engage in a conversation", "adjust the schedule", "buy a good"]:
				raise ValueError(f"Invalid react mode: {react_mode}. Use default react mode of continue doing current activity.")
			if react_mode == "buy a good" and react_target is None:
				raise ValueError(f"Invalid react target: {react_target} for buy a good. Use default react mode of continue doing current activity.")
		except Exception as e:
			self.logger.error(f"Error determining react mode: {e}. The response was {response}. Use default react mode of continue doing current activity.")
			react_mode = "continue doing current activity"
			react_target = None
			react_reason = ""
		self.logger.info(f"React mode: {react_mode}, target: {react_target}")
		return react_mode, react_target, react_reason

	def generate_commute_plan(self):
		try:
			if self.no_react:
				raise Exception("Do not revoke the llm in no react mode. Use default commute plan of walk.")
			prompt_template = open('agents/prompts/ella/prompt_commute.txt', 'r').read()
			prompt = prompt_template.replace("$Character$", self.get_character_description())
			prompt = prompt.replace("$State$", self.get_state_description())
			prompt = prompt.replace("$Time$", self.curr_time.strftime("%H:%M:%S"))
			prompt = prompt.replace("$Schedule$",
									json.dumps(self.hourly_schedule[self.curr_schedule_idx],
											   indent=2))
			prompt = prompt.replace("$Goal_place$", self.hourly_schedule[self.curr_schedule_idx]["end_place"])
			prompt = prompt.replace("$Transit$", json.dumps(self.s_mem.get_transit_schedule(self.curr_time), indent=2))
			prompt = prompt.replace("$EstimatedDistance1$", f"{self.get_estimated_distance_from_me_to_goal()}m")
			prompt = prompt.replace("$EstimatedDistance2$", f"{self.s_mem.get_estimated_distance_to_transit_station(self.pose[:2])}m")
			goal_place_dict = self.s_mem.get_knowledge(self.hourly_schedule[self.curr_schedule_idx]["end_place"])
			prompt = prompt.replace("$EstimatedDistance3$", f"{self.s_mem.get_estimated_distance_to_transit_station([goal_place_dict['location'][0], goal_place_dict['location'][1]])}m")
			self.logger.debug(f"Commute prompt: {prompt}")
			response = self.generator.generate(prompt, img=None, json_mode=False)
			self.logger.debug(f"Commute prompt response: {response}")
			commute_plan = self.parse_json(prompt, response)
			assert type(commute_plan) == list
			for commute in commute_plan:
				if commute["goal_place"] not in self.s_mem.places:
					raise Exception(f"Invalid goal place {commute['goal_place']} for commute.")
				if commute["transit_type"] != "walk":
					if commute["goal_place"] not in self.s_mem.get_transit_places():
						raise Exception(f"Invalid goal place {commute['goal_place']} for bus or bike.")
		except Exception as e:
			self.logger.error(f"Error generating commute plan. Use default commute plan of walk. The error was {e} with traceback: {traceback.format_exc()}.")
			commute_plan = [{
					"goal_place": self.hourly_schedule[self.curr_schedule_idx + 1]["place"],
					"transit_type": "walk"
				}]
		self.logger.debug(f"Commute plan: {commute_plan}")
		return commute_plan

	def generate_motion_schedule(self):
		"""
		Generate motion schedule based on the Avatar's character and desire.
		"""
		self.motion_place = self.current_place
		prompt_template = open('agents/prompts/ella/prompt_motion_schedule.txt', 'r').read()
		prompt = prompt_template.replace("$Character$", self.get_character_description())
		prompt = prompt.replace("$Plan$", str(self.hourly_schedule[self.curr_schedule_idx]))
		prompt = prompt.replace("$Memory$", self.get_object_description())
		self.logger.debug(f"motion schedule prompt: {prompt}")
		response = self.generator.generate(prompt, img=None, json_mode=False)
		try:
			motion_schedule = self.parse_json(prompt, response)
			
			self.logger.info(f"Motion schedule before process: {motion_schedule}")
			error_messages = motion_schedule_processor(motion_schedule)
			if len(error_messages) > 0:
				error_messages_verbalize = f"There are error messages for {self.name}'s schedule: \n" + ''.join(error_messages) + '\n'
				raise Exception(error_messages_verbalize)
			self.logger.info(f"Motion schedule generation for {self.name} is success!")
			self.logger.info(f"Motion schedule: {motion_schedule}")
		except Exception as e:
			self.logger.error(f"Error generating daily plan: {e} with traceback: {traceback.format_exc()}. The response was {response}")
			motion_schedule = {
				"description": "Wait until another activity",
				"time": self.hourly_schedule[self.curr_schedule_idx]['start_time'],
				"motion_list": [
					{
						"type": "wait"
					}
				]
			}

		return motion_schedule

	def generate_adjusted_schedule(self, react_reason, retrieved_events):
		prompt_template = open('agents/prompts/ella/prompt_adjust_schedule.txt', 'r').read()
		prompt = prompt_template.replace("$Character$", self.get_character_description())
		prompt = prompt.replace("$Schedule$", json.dumps(self.hourly_schedule, indent=2))
		prompt =prompt.replace("$Time$", self.curr_time.strftime("%H:%M:%S"))
		prompt = prompt.replace("$Place$", "None" if self.current_place is None else self.current_place)
		prompt = prompt.replace("$Context$", self.describe_events(retrieved_events))
		prompt = prompt.replace("$Reason$", react_reason)
		self.logger.debug(f"Adjust schedule prompt: {prompt}")
		response = self.generator.generate(prompt, img=None, json_mode=False)
		try:
			adjusted_schedule = self.parse_json(prompt, response)
			error_messages = schedule_validator(adjusted_schedule, self.s_mem, self.curr_time, logger=self.logger)
			if len(error_messages) > 0:
				error_messages_verbalize = f"There are error messages for {self.name}'s schedule: \n" + '\n'.join(error_messages) + '\n'
				raise Exception(error_messages_verbalize)
			self.logger.info(f"Adjusted schedule generation is success!")
		except Exception as e:
			self.logger.error(f"Error generating adjusted_schedule: {e} with traceback: {traceback.format_exc()}. The response was {response}")
			adjusted_schedule = None

		return adjusted_schedule

	def generate_hourly_schedule(self):
		"""EXAMPLE OUTPUT
		[
			{
				"start_time": "00:00:00",
				"end_time": "08:00:00",
				"activity": "Sleep"
				"place": "Home"
			},
			{
				"start_time": "09:00:00",
				"end_time": "09:30:00",
				"activity": "Commute"
				"place": null
			}
		]
		"""
		prompt_template = open('agents/prompts/ella/prompt_daily_plan.txt', 'r').read()
		prompt = prompt_template.replace("$Character$", self.get_character_description())
		prompt =prompt.replace("$Places$", self.get_places_description())
		prompt = prompt.replace("$Date$", self.get_curr_date())
		self.logger.debug(f"Daily plan prompt: {prompt}")
		response = self.generator.generate(prompt, img=None, json_mode=False)
		try:
			hourly_schedule = self.parse_json(prompt, response)
			error_messages = schedule_validator(hourly_schedule, self.s_mem, self.curr_time, logger=self.logger)
			if len(error_messages) > 0:
				error_messages_verbalize = f"There are error messages for {self.name}'s schedule: \n" + '\n'.join(error_messages) + '\n'
				raise Exception(error_messages_verbalize)
			self.logger.info(f"Hourly schedule generation for {self.name} is success!")
			self.logger.debug(f"Hourly schedule: {hourly_schedule}")
		except Exception as e:
			self.logger.error(f"Error generating daily plan: {e} with traceback: {traceback.format_exc()}. Using default instead.The response was {response}")
			# use default
			food_places = []
			work_shop_entertainment_places = []
			for place in self.s_mem.get_places():
				place_knowledge = self.s_mem.get_knowledge(place)
				if place_knowledge["coarse_type"] == "food":
					food_places.append(place)
				if place_knowledge["coarse_type"] in ["entertainment", "office", "stores"]:
					work_shop_entertainment_places.append(place)
			default_hourly_schedule = open('agents/prompts/ella/default_hourly_schedule.txt', 'r').read()
			default_hourly_schedule = default_hourly_schedule.replace("$name$", self.name)
			default_hourly_schedule = default_hourly_schedule.replace("$home$", self.scratch["living_place"])
			default_hourly_schedule = default_hourly_schedule.replace("$home_building$", self.s_mem.get_building_from_place(self.scratch["living_place"]))
			lunch_place = random.choice(food_places)
			default_hourly_schedule = default_hourly_schedule.replace("$lunch$", lunch_place)
			default_hourly_schedule = default_hourly_schedule.replace("$lunch_building$", self.s_mem.get_building_from_place(lunch_place))
			dinner_place = random.choice(food_places)
			default_hourly_schedule = default_hourly_schedule.replace("$dinner$", dinner_place)
			default_hourly_schedule = default_hourly_schedule.replace("$dinner_building$", self.s_mem.get_building_from_place(dinner_place))
			main1 = random.choice(work_shop_entertainment_places)
			default_hourly_schedule = default_hourly_schedule.replace("$main1$", main1)
			default_hourly_schedule = default_hourly_schedule.replace("$main1_building$", self.s_mem.get_building_from_place(main1))
			main2 = random.choice(work_shop_entertainment_places)
			default_hourly_schedule = default_hourly_schedule.replace("$main2$", main2)
			default_hourly_schedule = default_hourly_schedule.replace("$main2_building$", self.s_mem.get_building_from_place(main2))
			main3 = random.choice(work_shop_entertainment_places)
			default_hourly_schedule = default_hourly_schedule.replace("$main3$", main3)
			default_hourly_schedule = default_hourly_schedule.replace("$main3_building$", self.s_mem.get_building_from_place(main3))
			if '$' not in default_hourly_schedule:
				hourly_schedule = json.loads(default_hourly_schedule.strip())
			else:
				self.logger.error("Critical Error: '$' exists in the default hourly schedule")
				exit()
		# self.e_mem.add_memory("thoughts", self.curr_time, self.pose[:3], self.current_place, ["schedule"], None, f"Generated the hourly schedule.", None)
		self.last_adjust_schedule_time = self.curr_time
		return hourly_schedule

	def parse_json(self, prompt, response, last_call=False):
		json_str = None
		if "```json" in response:
			# Step 1: Extract the JSON part
			start = response.find("```json") + len("```json")
			end = response.find("```", start)
			json_str = response[start:end].strip()
		else:
			self.logger.warning(f"Error parsing JSON, the string was {response}")
			if not last_call:
				chat_history = [
					{"role": "user", "content": prompt},
					{"role": "assistant", "content": response}
				]
				data = self.generator.generate(
					f"The output format is wrong. Output the formatted json string enclosed in ```json``` only! Do not include any other character in the output!", chat_history=chat_history)
				return self.parse_json(None, data, last_call=True)
			else:
				self.logger.error(f"Error parsing JSON, already last call, the string was {response}")
				return None

		# # Step 2: Clean up the JSON
		# # Replace single quotes with double quotes
		# # Safely evaluate the string to a Python dictionary
		# parsed_dict = ast.literal_eval(json_str)
		# # Convert the dictionary back to a JSON string
		# json_str = json.dumps(parsed_dict)

		# Step 3: Convert to dictionary
		try:
			response = json.loads(json_str)
		except json.JSONDecodeError as e:
			self.logger.warning(f"Error decoding JSON: {e}, the string was {json_str}")
			if not last_call:
				chat_history = [
					{"role": "user", "content": prompt},
					{"role": "assistant", "content": response}
				]
				data = self.generator.generate(
					f"The output format is wrong. Output the formatted json string enclosed in ```json``` only! Do not include any other character in the output!", chat_history=chat_history)
				return self.parse_json(None, data, last_call=True)
		return response

	def describe_events(self, events):
		if events is None:
			return "No events."
		desc = ""
		for event in events:
			desc += f"type: {event['event_type']}\ntime: {event['event_time']}\nplace: {round_numericals(event['event_place'])}\nkeywords: {event['event_keywords']}\ncontent: {event['event_description']}\n\n"
		return desc

	def describe_knowledge(self, knowledge):
		if knowledge is None:
			return "No knowledge."
		desc = ""
		for key, value in knowledge.items():
			if ".png" in str(value):
				continue
			if "_idx" in key:
				continue
			if value is None:
				continue
			desc += f"{key}: {round_numericals(value)}\n"
		desc = f"{{\n{desc}}}\n"
		return desc

	def knowledge_name_to_items(self, knowledge_names: list[str]):
		items = []
		for knowledge_name in knowledge_names:
			knowledge = self.s_mem.get_knowledge(knowledge_name)
			if knowledge is None:
				self.logger.error(f"No knowledge found for {knowledge_name}.")
				continue
			item = {"name": knowledge_name}
			for key, value in knowledge.items():
				if ".png" in str(value):
					continue
				if "_idx" in key:
					continue
				if ".json" in str(value):
					continue
				if key == "bounding_box":
					continue
				item[key] = round_numericals(value)
			items.append(item)

		return json.dumps(items, indent=2)

	def get_curr_schedule_idx(self):
		if self.curr_time is None or len(self.hourly_schedule) == 0:
			return 0
		for i, schedule in enumerate(self.hourly_schedule):
			if datetime.strptime(schedule["start_time"], "%H:%M:%S").time() <= self.curr_time.time() <= datetime.strptime(schedule["end_time"], "%H:%M:%S").time():
				return i
		self.logger.error(f"Error finding current schedule index for {self.name} at {self.curr_time}. The hourly schedule is {self.hourly_schedule}.")
		return 0

	def get_places_description(self):
		places = []
		for place in self.s_mem.get_places():
			place_dict = self.s_mem.get_knowledge(place)
			if place_dict["coarse_type"] == "transit":
				continue
			places.append({"name": place, "type": place_dict["coarse_type"], "building": place_dict["building"]})
		return json.dumps(places, indent=2)

	def get_curr_date(self):
		if self.curr_time is None:
			return None
		return self.curr_time.strftime("%A %B %d")

	def get_state_description(self):
		if self.current_vehicle is not None:
			return f"I am riding a {self.current_vehicle} in the open space."
		if self.current_place is not None:
			return f"I am at {self.current_place}."
		return "I am walking in the open space."

	def get_object_description(self):
		descrption = "["
		for obj in self.s_mem.scene_graph_dict[self.s_mem.current_place].objects.values():
			descrption += f'''
{{
	'name' : '{obj.name+str(obj.idx)}',
}}
				'''
		descrption += "]"
		return descrption
	
	def get_character_description(self):
		"""EXAMPLE OUTPUT
		   Name: Dolores Heitmiller
		   Age: 28
		   Innate traits: hard-edged, independent, loyal
		   Learned traits: Dolores is a painter who wants live quietly and paint
			 while enjoying her everyday life.
		   Currently: Dolores is preparing for her first solo show. She mostly
			 works from home.
		   Lifestyle: Dolores goes to bed around 11pm, sleeps for 7 hours, eats
			 dinner around 6pm.
			Groups:
		   Daily plan requirement:
		   Current Date: Monday, January 1
		"""
		return f"""Name: {self.name}
Age: {self.scratch['age']}
Innate traits: {self.scratch['innate']}
Learned traits: {self.scratch['learned']}
Currently: {self.scratch['currently']}
Lifestyle: {self.scratch['lifestyle']}
Groups: {self.scratch['groups']}
Daily plan requirement: {self.scratch['daily_requirement']}
Held objects: {self.held_objects}
Cash: {self.cash}
Current date: {self.get_curr_date()}"""

	def get_closest_bus_stop(self, pos):
		closest_stop = None
		min_dist = float('inf')
		for transit_place_name in self.s_mem.get_transit_places():
			transit_place = self.s_mem.get_knowledge(transit_place_name)
			if transit_place["fine_type"] != "bus stop":
				continue
			dist = np.linalg.norm(np.array(transit_place["location"]) - np.array(pos))
			if dist < min_dist:
				min_dist = dist
				closest_stop = transit_place_name
		return closest_stop

	def get_closest_bike_station(self, pos):
		closest_station = None
		min_dist = float('inf')
		for transit_place_name in self.s_mem.get_transit_places():
			transit_place = self.s_mem.get_knowledge(transit_place_name)
			if transit_place["fine_type"] != "bike station":
				continue
			dist = np.linalg.norm(np.array(transit_place["location"]) - np.array(pos))
			if dist < min_dist:
				min_dist = dist
				closest_station = transit_place_name
		return closest_station

	def get_estimated_distance_from_me_to_goal(self):
		if self.hourly_schedule[self.curr_schedule_idx]["type"] != "commute":
			return ""
		my_pos = self.pose[:2]
		goal_place = self.hourly_schedule[self.curr_schedule_idx]["end_place"]
		goal_place_dict = self.s_mem.get_knowledge(goal_place)
		if goal_place_dict is None:
			self.logger.error(f"No knowledge found for {goal_place}.")
			return ""
		goal_pos = np.array([goal_place_dict["location"][0], goal_place_dict["location"][1]])
		estimated_distance = int(np.linalg.norm(np.array(goal_pos) - np.array(my_pos)))
		return estimated_distance

	def save_scratch(self):
		self.scratch['hourly_schedule'] = self.hourly_schedule
		self.scratch['commute_plan'] = self.commute_plan
		self.scratch['commute_plan_idx'] = self.commute_plan_idx
		self.scratch['last_enter_bus_time'] = self.last_enter_bus_time.strftime("%B %d, %Y, %H:%M:%S") if self.last_enter_bus_time is not None else None
		self.scratch['chatting_buffer'] = [chat.to_dict() for chat in self.chatting_buffer]
		super().save_scratch()

if __name__ == '__main__':
	config = json.load(open("ViCo/assets/scenes/DETROIT/agents_num_15_with_schedules/config.json", "r"))
	agent_id = 0
	from vico.agents import AgentLogger
	# global_model_manager.init(local=True)
	logger = AgentLogger(config["agent_names"][agent_id], "info","a.log")

	agent = EllaAgent(name=config["agent_names"][agent_id], pose=config["agent_poses"][agent_id], info=config["agent_infos"][agent_id],
					  sim_path="ViCo/assets/scenes/DETROIT/agents_num_15_with_schedules", no_react=False, debug=True, logger=logger, lm_source='azure', lm_id='gpt-4o',
					  detect_interval=-1)
	# agent.scratch["daily_requirement"] = f"My group {agent.scratch['groups'][0]['name']} is organizing a party at {agent.scratch['groups'][0]['place']} from 14:30:00 to 15:00:00 today. I need to go around the city, find and invite people outside of my group to attend our party today."
	# agent.scratch["currently"] = agent.scratch["daily_requirement"]
	# agent.hourly_schedule = agent.generate_hourly_schedule()
	# adjusted_schedule = agent.generate_adjusted_schedule("There is no person at the place.", [
	# 	{
	# 		"event_type": "conversation",
	# 		"event_time": "08:50:00",
	# 		"event_place": "Open Space",
	# 		"event_keywords": ["conversation", "someone"],
	# 		"event_description": "taylor invited to me attent their group's party at Pulse Music from 8:30 PM to 9:00 PM."
	# 	}])
	agent.set_curr_time(datetime.strptime("2022-01-01 15:00:01", "%Y-%m-%d %H:%M:%S"))
	print(agent.hourly_schedule[agent.curr_schedule_idx])
	agent.generate_commute_plan()
	# agent.parse_json("test", "\"option\": \"engage in a conversation\", \"target\": \"someone\"", last_call=False)
	# with open(f"ViCo/assets/scenes/DETROIT/agents_test_react_num_2/hourly_schedule_{config['agent_names'][agent_id]}.json", "w") as f:
	# 	json.dump(hourly_schedule, f, indent=2)