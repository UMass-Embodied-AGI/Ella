import os, sys
from datetime import time

from tools.model_manager import global_model_manager
from .sg.builder.object import AGENT_TAGS
from .gen_agent_memory import SemanticMemory, EpisodicMemory
_ella_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ella_root)
sys.path.insert(0, os.path.join(_ella_root, 'vico'))
from vico.agents import Agent
from vico.tools.utils import *

class GenAgent(Agent):
	def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None, lm_source='azure', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0, enable_gt_segmentation=True,
				 model_channel=None, model_device=None):
		super().__init__(name, pose, info, sim_path, no_react, debug, logger)
		if model_channel is not None:
			global_model_manager.set_channel(model_channel, model_device)
		self.lm_source = lm_source
		self.wake_up_hour = None
		self.generator = global_model_manager.get_generator(lm_source, lm_id, max_tokens, temperature, top_p, logger)
		self.generator_embedding = global_model_manager.get_generator(lm_source, 'text-embedding-3-small', max_tokens, temperature, top_p, logger)
		self.react_freq = self.scratch["react_freq"] if "react_freq" in self.scratch else 60
		self.generate_new_schedule_freq = 600
		self.last_new_schedule_time = None
		self.importance_trigger_curr = self.scratch["importance_trigger_curr"]
		self.importance_trigger_max = self.scratch["importance_trigger_max"]
		self.importance_ele_n = self.scratch["importance_ele_n"]
		self.daily_plan = self.scratch["daily_plan"]
		self.hourly_schedule = self.scratch["hourly_schedule"]
		self.curr_event = self.scratch["act_event"]
		self.last_actions = [None] * 5 # maintain 5 last actions
		self.enable_gt_segmentation = enable_gt_segmentation
		self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), detect_interval=-1 if self.enable_gt_segmentation else 2, fov=self.fov, debug=self.debug, logger=self.logger)
		self.e_mem = EpisodicMemory(os.path.join(self.storage_path, "episodic_memory"), lm_source, debug=self.debug, logger=self.logger)
		lastevent = self.e_mem.get_memory(-1)
		if lastevent is not None:
			self.last_react_time = lastevent.event_time
		else:
			self.last_react_time = None
		self.commuting = None
		if self.scratch["act_address"] is None:
			self.curr_goal_address = None
			self.curr_goal_pos = None
			self.curr_goal_bbox = None
			self.curr_goal_description = None
			self.curr_goal_duration = None
			self.curr_goal_end_time = None
		else:
			curr_goal_dict = self.s_mem.get_knowledge(self.scratch["act_address"])
			self.curr_goal_address = self.scratch["act_address"]
			self.curr_goal_pos = np.array([curr_goal_dict["location"][0], curr_goal_dict["location"][1]])
			self.curr_goal_bbox = np.array(curr_goal_dict["bounding_box"])
			self.curr_goal_description = self.scratch["act_description"]
			self.curr_goal_duration = self.scratch["act_duration"]
			if "act_start_time" in self.scratch and self.scratch["act_start_time"] is not None:
				self.curr_goal_end_time = datetime.strptime(self.scratch["act_start_time"], "%B %d, %Y, %H:%M:%S") + timedelta(minutes=self.curr_goal_duration)
			else:
				next_schedule_index = self.get_next_schedule_index()
				curr_goal_start_time = datetime.combine(self.curr_time.date(), datetime.min.time()) + timedelta(minutes=sum(activity[1] for activity in self.hourly_schedule[:next_schedule_index]))
				self.curr_goal_end_time = curr_goal_start_time + timedelta(minutes=self.curr_goal_duration)
		self.need_react_in_building = True

	def update_action(self, action):
		self.last_actions.pop(0)
		self.last_actions.append(action)

	def reset(self, name, pose):
		super().reset(name, pose)
		if self.scratch["act_address"] is None:
			self.curr_goal_pos = None
			self.curr_goal_bbox = None
		else:
			curr_goal_dict = self.s_mem.get_knowledge(self.scratch["act_address"])
			self.curr_goal_pos = np.array([curr_goal_dict["location"][0], curr_goal_dict["location"][1]])
			self.curr_goal_bbox = np.array(curr_goal_dict["bounding_box"])
		self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), detect_interval=-1 if self.enable_gt_segmentation else 2, fov=self.fov, debug=self.debug, logger=self.logger)
		self.e_mem = EpisodicMemory(os.path.join(self.storage_path, "episodic_memory"), self.lm_source, debug=self.debug, logger=self.logger)

	def _process_obs(self, obs): # override
		self.cur_objects = self.s_mem.update_ga(obs)
		self.current_place = obs['current_place']
		self.obs = obs
		self.curr_time = obs['curr_time']
		if self.obs['new_day'] or self.hourly_schedule == []:
			if len(self.e_mem) == 0:
				self.daily_plan = self.generate_daily_plan()
			else:
				self.daily_plan = self.generate_daily_plan()
				# self.daily_plan = self.revise_identity_and_generate_daily_plan()
			self.scratch["daily_plan"] = self.daily_plan
			self.logger.debug(f"{self.name}'s daily plan is {self.daily_plan}.")
			self.hourly_schedule = self.generate_hourly_schedule()

			self.logger.info(f"{self.name}'s hourly schedule is {self.hourly_schedule}.")
		self.perceived = []

		if len(self.obs['events']) > 0:
			desc = ""
			for event in self.obs['events']:
				img_path = None
				kws = []
				if event["type"] == "speech":
					kws = []
					if event["position"][:2] == self.pose[:2]:
						continue
					if event["subject"] is None:
						event["subject"] = self.s_mem.get_name_from_position(event["position"])
						kws.append(event["subject"])
					if event["subject"] is not None:
						event["content"] = {"from_name": event["subject"],"utterance": event["content"]}
						desc = event["content"]["from_name"] + " said '" + event["content"]["utterance"] + "'"
						kws.append(event["subject"])
					else:
						desc = ""
				elif event["type"] == "store goods":
					desc = f"I saw a shelf with a variety of goods ({event['content']}) in a store"
					kws = re.findall(r"'name': '([^']*)'", event["content"])
				else:
					desc = ""
				if desc != "" and len(kws) > 0:
					if desc not in self.e_mem.experience_embedding.keys():
						embedding = self.generator_embedding.get_embedding(desc, caller="embed_speech_description")
					else:
						embedding = self.e_mem.experience_embedding[desc]
					self.perceived.append(self.e_mem.add_memory(event_type=event["type"], event_time=self.curr_time, event_position=event["position"], event_place=self.obs['current_place'], event_keywords=kws, event_img=img_path, event_description=desc, event_embedding=embedding, event_poignancy=self.generate_poig_score(desc)))
			if desc != "":
				self.last_react_time = self.curr_time
		
		if self.obs['rgb'] is not None:
			bool1 = self.last_react_time is None
			bool2 = (self.last_react_time != self.curr_time and (self.curr_time - self.last_react_time).total_seconds() % self.react_freq == 0) if bool1 == False else None
			bool3 = ((self.last_actions[-1] is not None and self.last_actions[-1]["type"] == "turn_around_to_sound_source") or (self.last_actions[-2] is not None and self.last_actions[-2]["type"] == "turn_around_to_sound_source")) if bool1 == False and bool2 == False else None
			bool4 = (len(self.e_mem.curr_chat) > 0) if bool1 == False and bool2 == False and bool3 == False else None
			if bool1 or bool2 or bool3 or bool4:
				if self.enable_gt_segmentation:
					obs_seg_unique_ids = np.unique(self.obs['segmentation']).tolist()
					img_path = None
					for seg_id in obs_seg_unique_ids:
						if seg_id != 0:
							seg_id_count = np.count_nonzero(self.obs['segmentation'] == seg_id)
							if (seg_id_count / np.array(self.obs['segmentation']).size) > 0.001:
								seg_info = self.obs["gt_seg_idxc_to_info"][seg_id]
								seg_type = seg_info["type"]
								seg_name = seg_info["name"]
								kws = [seg_type + ':' + seg_name, "", ""]
								desc = f"I saw {seg_name}"
								if desc not in self.e_mem.experience_embedding.keys():
									embedding = self.generator_embedding.get_embedding(desc, caller="embed_observation_description")
								else:
									embedding = self.e_mem.experience_embedding[desc]
								self.perceived.append(self.e_mem.add_memory(event_type="observation", event_time=self.curr_time, event_position=self.pose[:3], event_place=self.obs['current_place'], event_keywords=kws, event_img=img_path, event_description=desc, event_embedding=embedding, event_poignancy=self.generate_poig_score(desc)))
				else:
					img_path = None
					for cur_obj in self.cur_objects:
						seg_type = "avatar" if cur_obj["tag"] in AGENT_TAGS else cur_obj["tag"]
						seg_name = cur_obj["name"]
						kws = [seg_type + ':' + seg_name, "", ""]
						desc = f"I saw {seg_name}"
						if desc not in self.e_mem.experience_embedding.keys():
							embedding = self.generator_embedding.get_embedding(desc, caller="embed_observation_description")
						else:
							embedding = self.e_mem.experience_embedding[desc]
						self.perceived.append(self.e_mem.add_memory(event_type="observation", event_time=self.curr_time, event_position=self.pose[:3], event_place=self.obs['current_place'], event_keywords=kws, event_img=img_path, event_description=desc, event_embedding=embedding, event_poignancy=self.generate_poig_score(desc)))
				self.last_react_time = self.curr_time
				self.e_mem.save_memory()
		for event in self.perceived:
			self.importance_trigger_curr -= event.event_poignancy
			self.importance_ele_n += 1
		if self.obs['action_status'] == "FAIL":
			self.logger.info(f"{self.name} failed to execute last action.")
			if self.last_actions[-1] is not None:
				if self.last_actions[-1]["type"] == 'converse':
					print(f"{self.name} action status is fail so dropping the following chat: {self.e_mem.curr_chat[-1]}")
					self.e_mem.curr_chat.pop()
					self.update_action(None)
	
	def get_focused_event(self, retrieved):
		copy_retrieved = retrieved.copy()

		for index, retrieved_single in enumerate(copy_retrieved): 
			curr_event = retrieved_single["curr_event"]
			if curr_event.event_keywords[0] == self.name:
				retrieved.pop(index)
		
		priority = []
		for retrieved_single in retrieved: 
			curr_event = retrieved_single["curr_event"]
			if (curr_event.event_type == "observation" and curr_event.event_keywords[0].split(':')[0] == "avatar"):
				priority += [retrieved_single]
		if priority:
			return random.choice(priority)

		for retrieved_single in retrieved: 
			curr_event = retrieved_single["curr_event"]
			if "is idle" not in curr_event.event_description: 
				priority += [retrieved_single]
		if priority:
			return random.choice(priority)
		
		return None
	
	def reflection_trigger(self):
		if self.importance_trigger_curr <= 0 and [] != self.e_mem.experience:
			return True
		return False
	
	def generate_focal_points(self, num=3):
		events = [[event.event_last_access_time, event] for event in self.e_mem.experience if "idle" not in event.event_description]
		events = sorted(events, key=lambda x: x[0])
		events = [event for _, event in events]

		statements = ""
		for event in events[-1*self.importance_ele_n:]: 
			statements += event.event_description + "\n"
		
		with open ("agents/prompts/gen_agent/v3_ChatGPT/generate_focal_pt_v1.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", statements)
		prompt = prompt.replace("!<INPUT 1>!", str(num))
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		example_output = '["What should Jane do for lunch", "Does Jane like strawberry", "Who is Jane"]'
		special_instruction = "Output must be a list of str."
		response = self.generator.generate(self.compose_instruction_prompts(prompt=prompt, example_output=example_output, special_instruction=special_instruction), caller="generate_focal_points")
		try:
			response = "1) " + response.strip()
			ret = []
			for i in response.split("\n"): 
				ret += [i.split(") ")[-1]]
			return ret
		except:
			return ["Who am I"] * num
		
	def generate_insights_and_evidence(self, events, num=5):
		statements = ""
		for count, event in enumerate(events): 
			statements += f'{str(count)}. {event["event_description"]}\n'
		
		with open ("agents/prompts/gen_agent/v2/insight_and_evidence_v1.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", statements)
		prompt = prompt.replace("!<INPUT 1>!", str(num))
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		response = self.generator.generate(prompt, caller="generate_insights_and_evidence")
		response = "1. " + response.strip()
		ret = dict()
		for i in response.split("\n"): 
			row = i.split(". ")[-1]
			try:
				thought = row.split("(because of ")[0].strip()
				evi_raw = row.split("(because of ")[1].split(")")[0].strip()
				evi_raw = re.findall(r'\d+', evi_raw)
				evi_raw = [int(i.strip()) for i in evi_raw]
				ret[thought] = evi_raw
			except:
				continue
		try:
			for thought, evi_raw in ret.items(): 
				evidence_node_id = [events[i].event_id for i in evi_raw]
				ret[thought] = evidence_node_id
				if len(thought.strip()) == 0:
					ret.pop(thought)
			return ret
		except:
			return {"this is blank": "node_1"}
		
	def generate_poig_score(self, description):
		if "is idle" in description:
			return 1
		
		with open ("agents/prompts/gen_agent/v3_ChatGPT/poignancy_event_v1.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", self.name)
		prompt = prompt.replace("!<INPUT 1>!", self.get_character_description())
		prompt = prompt.replace("!<INPUT 2>!", self.name)
		prompt = prompt.replace("!<INPUT 3>!", description)
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		example_output = "5"
		special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10."
		response = self.generator.generate(self.compose_instruction_prompts(prompt=prompt, example_output=example_output, special_instruction=special_instruction), caller="generate_poig_score")
		try:
			response = response.strip()
			return int(response)
		except:
			return 4
	
	def reflect(self):
		focal_points = self.generate_focal_points(num=3)
		focal_points_embedding = []
		for focal_point in focal_points:
			if focal_point != "":
				focal_points_embedding.append(self.generator_embedding.get_embedding(focal_point, caller="embed_focal_point") if focal_point not in self.e_mem.experience_embedding.keys() else self.e_mem.experience_embedding[focal_point])
		retrieved = self.e_mem.new_retrieve(self.curr_time, focal_points, focal_points_embedding, 30)
		for focal_point, events in retrieved.items():
			thoughts = self.generate_insights_and_evidence(events, 5)
			for thought, evidence in thoughts.items():
				expiration = self.curr_time + timedelta(days=30)
				s, p, o = self.generate_action_event_triple(thought)
				keywords = [s, p, o]
				thought_poignancy = self.generate_poig_score(thought)
				if thought not in self.e_mem.experience_embedding.keys():
					if thought != "":
						embedding = self.generator_embedding.get_embedding(thought, caller="embed_thought")
					else:
						continue
				else:
					embedding = self.e_mem.experience_embedding[thought]
				img_path = None
				self.e_mem.add_memory(event_type="thought", event_time=self.curr_time, event_position=self.pose[:3], event_place=self.obs['current_place'], event_keywords=keywords, event_img=img_path, event_description=thought, event_embedding=embedding, event_poignancy=thought_poignancy, event_expiration=expiration)

	def reset_reflection_counter(self):
		self.importance_trigger_curr = self.importance_trigger_max
		self.importance_ele_n = 0
	
	def last_actions_handler(self):
		if self.last_actions[-1] is not None and self.last_actions[-1]["type"] == 'converse':
			return None
		
		if self.last_actions[-1] is not None and self.last_actions[-1]["type"] == "turn_around_to_sound_source": # here, we assume the sound source must be an avatar
			has_avatar_in_perception = False
			for event_instance in self.perceived:
				if event_instance.event_type == "observation" and "avatar" in event_instance.event_keywords[0]:
					has_avatar_in_perception = True
			if not has_avatar_in_perception:
				print("Turned around but not finding any avatar, hence walking 5m forward")
				action = {
					'type': 'move_forward',
					'arg1': 5,
				}
				return action
		return "continue"
	
	def obs_events_handler(self):
		for event in self.obs["events"]:
			if event["type"] == "speech" and event["content"]["from_name"] != self.name:
				sound_source_in_perception = False
				for event_instance in self.perceived:
					if event_instance.event_type == "observation" and event['content']['from_name'] in event_instance.event_keywords[0]:
						sound_source_in_perception = True
				self.e_mem.curr_chat += [[event["content"]["from_name"], event["content"]["utterance"]]]
				if not sound_source_in_perception:
					action = self.turn_to_pos(event["position"])
					action_temp = action.copy()
					action_temp['type'] = "turn_around_to_sound_source"
					return action
		return "continue"
	
	def focused_event_handler(self, focused_event):
		if focused_event["curr_event"].event_type == "observation" and focused_event["curr_event"].event_keywords[0].split(':')[0] == "avatar" and focused_event["curr_event"].event_keywords[0].split(':')[1] != self.name: # focused_event.event_keywords[0] is the subject of the observation (type:name)
			target_knowledge = self.s_mem.get_knowledge(focused_event["curr_event"].event_keywords[0].split(':')[1])
			if self.react_converse(self.obs, focused_event, target_knowledge):
				self.need_react_in_building = True
				experience = self.e_mem.retrieve_latest_memory()
				observation_kws = None
				for event in experience:
					if event["event_type"] == "observation":
						observation_kws = event["event_keywords"]
				if observation_kws == None:
					raise ValueError("observation_kws should not be None, there's a bug somewhere")
				utt, _ = self.converse(self.obs, focused_event, target_agent_state="null")
				if self.enable_gt_segmentation:
					entity_idx_list = []
					for idx, info in enumerate(self.obs["gt_seg_idxc_to_info"]):
						if f"{info['type']}:{info['name']}" == focused_event["curr_event"].event_keywords[0]:
							entity_idx_list.append(idx)
					assert len(entity_idx_list) > 0
					mask_of_obj = np.isin(self.obs["segmentation"], entity_idx_list)
					if mask_of_obj.any():
						obj_coords = np.argwhere(mask_of_obj)
						centroid = np.mean(obj_coords, axis=0)
						cy, cx = centroid.astype(int)
						center_depth = self.obs["depth"][cy, cx]
					converse_range = center_depth + 3
				else:
					target_agent_name = focused_event["curr_event"].event_keywords[0].split('avatar:')[1]
					target_position = None
					for cur_obj in self.cur_objects:
						if cur_obj["name"] == target_agent_name:
							print(f"target agent {cur_obj['name']} DISTANCE:", np.linalg.norm(np.array(cur_obj["position"]) - np.array(self.pose[:3])))
							target_position = cur_obj["position"]
							break
					converse_range = np.linalg.norm(np.array(target_position) - np.array(self.pose[:3])) + 1
					print("converse range:", converse_range)
				action = {
					'type': 'converse',
					'arg1': utt,
					'arg2': converse_range
				}
				print(f"{self.name}: {utt}")

				return action
		elif focused_event["curr_event"].event_type == "store goods":
			item = self.react_buy_good(focused_event)
			if item is not None:
				if self.held_objects[0] is None:
					action = {
						'type': 'pick',
						'arg1': 0,
						'arg2': item
					}
					self.update_action(action)
					desc = f"Pick up {item} at {self.current_place}."
					if desc not in self.e_mem.experience_embedding.keys():
						embedding = self.generator_embedding.get_embedding(desc, caller="embed_speech_description")
					else:
						embedding = self.e_mem.experience_embedding[desc]
					self.e_mem.add_memory(event_type="action", event_time=self.curr_time, event_position=self.pose[:3], event_place=self.current_place, event_keywords=[item, "pick", ""], event_img=None, event_description=desc, event_embedding=embedding, event_poignancy=self.generate_poig_score(desc))
					return action
				elif self.held_objects[1] is None:
					action = {
						'type': 'pick',
						'arg1': 1,
						'arg2': item
					}
					self.update_action(action)
					desc = f"Pick up {item} at {self.current_place}."
					if desc not in self.e_mem.experience_embedding.keys():
						embedding = self.generator_embedding.get_embedding(desc, caller="embed_speech_description")
					else:
						embedding = self.e_mem.experience_embedding[desc]
					self.e_mem.add_memory(event_type="action", event_time=self.curr_time, event_position=self.pose[:3], event_place=self.current_place, event_keywords=[item, "pick", ""], event_img=None, event_description=desc, event_embedding=embedding, event_poignancy=self.generate_poig_score(desc))
					return action
				else:
					self.logger.warning(f"I already held two objects {self.held_objects}, cannot pick up {item}.")

		return "continue"
			
	def generate_new_decomp_schedule(self, react_summary):
		hour_str = [
			"00:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM",
			"05:00 AM", "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM",
			"10:00 AM", "11:00 AM", "12:00 PM", "01:00 PM", "02:00 PM",
			"03:00 PM", "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM",
			"08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"
		]
		
		def parse_time(s):
			hhmm, ap = s.split()
			hh, mm = map(int, hhmm.split(':'))
			if ap == 'AM' and hh == 12:
				hh = 0
			if ap == 'PM' and hh != 12:
				hh += 12
			return time(hh, mm)
		
		tlist = [parse_time(s) for s in hour_str]
		next_hour_idx = 0
		for i in range(len(tlist) - 1):
			if tlist[i] <= self.curr_time.time() < tlist[i+1]:
				next_hour_idx = i + 1
				break
		
		old_compressed = self.hourly_schedule
		old_hourly_tasks = []
		for task, duration in old_compressed:
			hours_for_task = duration // 60
			old_hourly_tasks.extend([task] * hours_for_task)
		
		new_hourly_tasks = []
		with open("agents/prompts/gen_agent/v2/generate_hourly_schedule_v2.txt", "r") as f:
			prompt_template = f.read()
		
		schedule_format = ""
		for h in hour_str:
			schedule_format += f"[{self.get_curr_date()} -- {h}] Activity: [Fill in]\n"
		schedule_format = schedule_format.strip()
		
		prior_schedule = ""
		if next_hour_idx > 0:
			prior_schedule = "\n"
			for count in range(next_hour_idx):
				prior_schedule += f" {self.get_curr_date()} -- {hour_str[count]}] Activity: {self.scratch['first_name']} is {old_hourly_tasks[count]}\n"
		
		intermission_str = f"Here is today's originally intended plan for {self.scratch['first_name']}: "
		for count, p in enumerate(self.daily_plan):
			intermission_str += f"{count+1}) {p}, "
		intermission_str = intermission_str.rstrip(", ")
		intermission_str += f". New info: {react_summary}"
		
		for h_idx in range(next_hour_idx, 24):
			curr_hour_str = hour_str[h_idx]
			prompt = prompt_template
			prompt = prompt.replace("!<INPUT 0>!", schedule_format)
			prompt = prompt.replace("!<INPUT 1>!", self.get_character_description())
			prompt = prompt.replace("!<INPUT 2>!", prior_schedule)
			prompt = prompt.replace("!<INPUT 3>!", intermission_str)
			prompt = prompt.replace("!<INPUT 4>!", "")
			prompt = prompt.replace("!<INPUT 5>!", f" {self.get_curr_date()} -- {curr_hour_str}] Activity: {self.scratch['first_name']} is")
			if "<commentblockmarker>###</commentblockmarker>" in prompt:
				prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
			response = self.generator.generate(prompt, caller="generate_new_decomp_schedule")
			cr = response.strip()
			try:
				if '[' in cr:
					cr = cr.split("[")[0].strip()
				if cr.endswith('.'):
					cr = cr[:-1]
			except:
				cr = "sleeping"
			new_hourly_tasks.append(cr)
			prior_schedule += f" {self.get_curr_date()} -- {curr_hour_str}] Activity: {self.scratch['first_name']} is {cr}\n"
		
		combined_hourly_tasks = old_hourly_tasks[:next_hour_idx] + new_hourly_tasks
		_compressed = []
		prev_task = None
		for i in range(len(combined_hourly_tasks)):
			combined_hourly_tasks[i] = combined_hourly_tasks[i].split('\n')[0] if '\n' in combined_hourly_tasks[i] else combined_hourly_tasks[i]
		
		for i in combined_hourly_tasks:
			if i != prev_task:
				_compressed.append([i, 1])
				prev_task = i
			else:
				_compressed[-1][1] += 1
		final_compressed = []
		for task, hour_count in _compressed:
			final_compressed.append([task, hour_count * 60])
		return final_compressed
	
	def _act(self, obs):

		if self.current_place == self.scratch["groups"][0]["place"]:
			if self.held_objects[0] is not None:
				action = {
					'type': 'put',
					'arg1': 0
				}
				self.update_action(action)
				self.logger.info(f"{self.name} is at the group place, put down the object in hand 0.")
				return action
			elif self.held_objects[1] is not None:
				action = {
					'type': 'put',
					'arg1': 1
				}
				self.update_action(action)
				self.logger.info(f"{self.name} is at the group place, put down the object in hand 1.")
				return action

		if self.last_actions[-1] is not None and self.last_actions[-1]["type"] == "enter":
			self.need_react_in_building = True
		if obs["current_place"] is not None or self.commuting == False:
			if self.curr_goal_address and (obs['curr_time'] != self.curr_goal_end_time):
				has_speech_event_in_building = False
				for event in self.obs["events"]:
					if event["type"] == "speech" and event["position"][:2] != self.pose[:2]:
						has_speech_event_in_building = True
				if (not has_speech_event_in_building) and (not self.need_react_in_building):
					return {
						'type': 'sleep'
					}
				else:
					self.last_react_time = self.curr_time

		# Do memory retrieval first
		retrieved = self.e_mem.retrieve_events_thoughts_by_keywords(self.perceived)
		focused_event = self.get_focused_event(retrieved) # this is used for react() only
		
		if not self.curr_goal_address or obs['curr_time'] == self.curr_goal_end_time:
			self.plan(obs["rgb"])

		if self.reflection_trigger():
			self.reflect()
			self.reset_reflection_counter()
			
		action = self.last_actions_handler()
		if action != "continue":
			self.update_action(action)
			return action

		action = self.obs_events_handler()
		if action != "continue":
			self.update_action(action)
			return action
		
		if self.last_react_time == self.curr_time:
			self.need_react_in_building = False
			if focused_event:
				action = self.focused_event_handler(focused_event)
				if action != "continue":
					self.update_action(action)
					return action
				
		if self.curr_goal_address == obs['current_place'] or self.commuting == False:
			self.logger.info(f"{self.name} at {self.curr_goal_address} is {self.scratch['act_description']} for {self.scratch['act_duration']} minutes.")
			self.update_action(None)
			return None

		if self.curr_goal_address in obs['accessible_places']:
			self.logger.info(f"{self.name} finished navigation to {self.curr_goal_address} at {self.curr_goal_pos}, begin to {self.curr_goal_description} for {self.curr_goal_duration} minutes.")
			action = {
				'type': 'enter',
				'arg1': self.curr_goal_address
			}
			self.update_action(action)
			self.commuting = False
			return action
		# at wrong place, need to enter open space
		if obs['current_place'] is not None:
			self.logger.info(f"{self.name} at {obs['current_place']} is entering open space to move to {self.curr_goal_address} at {self.curr_goal_pos}.")
			action = {
				'type': 'enter',
				'arg1': 'open space'
			}
			self.update_action(action)
			self.commuting = True
			return action
		# at open space, need to move to the correct place
		cur_trans = np.array(self.pose[:2])
		if is_near_goal(cur_trans[0], cur_trans[1], self.curr_goal_bbox, self.curr_goal_pos):
			self.logger.warning(f"{self.name} at {self.pose[:3]} is near the goal {self.curr_goal_pos}, but not at the goal {self.curr_goal_address}.")
			self.update_action(None)
			return None
		self.logger.info(f"{self.name} at {self.pose[:3]} is moving to {self.curr_goal_address} at {self.curr_goal_pos} to {self.curr_goal_description} for {self.curr_goal_duration} minutes.")
		nav_action = self.navigate()
		self.update_action(nav_action)
		return nav_action

	def navigate(self):
		return self.navigate_helper(self.s_mem.get_sg(self.current_place), self.curr_goal_pos, self.curr_goal_bbox)

	def navigate_helper(self, sg, goal_pos, goal_bbox=None):
		from vico.tools.utils import get_bbox, get_axis_aligned_bbox
		if goal_pos is None:
			return None
		cur_trans = np.array(self.pose[:2])
		goal_bbox = get_bbox(goal_bbox, goal_pos)
		path = sg.volume_grid_builder.navigate(cur_trans, goal_bbox, self.last_path)
		self.last_path = None
		if path is None:
			self.logger.error(f"No path found when navigate agent {self.name} to {goal_pos}.")
			return None
		else:
			if self.current_vehicle == "bicycle":
				nav_grid_num = int(self.BIKE_SPEED // sg.volume_grid_builder.conf.nav_grid_size)
			elif self.current_vehicle is None:
				nav_grid_num = int(self.WALK_SPEED // sg.volume_grid_builder.conf.nav_grid_size)
			else:
				self.logger.error(f"Unsupported vehicle type {self.current_vehicle} for navigation.")
				nav_grid_num = int(self.WALK_SPEED // sg.volume_grid_builder.conf.nav_grid_size)
			cur_goal = path[min(nav_grid_num, len(path) - 1)]
			if sg.volume_grid_builder.has_obstacle(get_axis_aligned_bbox(np.array([cur_goal, cur_trans]), None)):
				cur_goal = path[min(2, len(path) - 1)]
		
		self.logger.debug(f"Path {path[:3]}\n...\n{path[-3:]}")
		from .sg.builder.volume_grid import convex_hull, dist_to_hull
		dist = dist_to_hull(path[-1], convex_hull(goal_bbox))
		if dist > 2:
			self.logger.warning(f"Unable to find a path to the target bounding box. The optimal available path is still a distance of {dist} away from the target bounding box. The optimal path has been automatically adopted.")
		if self.action_status == "COLLIDE":
			self.logger.warning(f"{self.name} at {self.pose} moving to {cur_goal} is colliding with obstacles, path found was {path}.")
		# move
		target_rad = np.arctan2(cur_goal[1] - cur_trans[1], cur_goal[0] - cur_trans[0])
		delta_rad = target_rad - self.pose[-1]
		if delta_rad > np.pi:
			delta_rad -= 2 * np.pi
		elif delta_rad < -np.pi:
			delta_rad += 2 * np.pi

		if delta_rad > np.deg2rad(15):
			action = {
				'type': 'turn_left',
				'arg1': np.rad2deg(delta_rad),
			}
			self.last_path = path
		elif delta_rad < -np.deg2rad(15):
			action = {
				'type': 'turn_right',
				'arg1': np.rad2deg(-delta_rad),
			}
			self.last_path = path
		else: action = {
			'type': 'move_forward',
			'arg1': np.linalg.norm(cur_goal - cur_trans),
		}
		if action['arg1'] < 0.1:
			self.logger.warning(f"{self.name} at {self.pose} moving to {cur_goal} is too close, path found was {path}.")
		return action

	def plan(self, rgb):
		next_schedule_index = self.get_next_schedule_index()
		self.curr_goal_description, self.curr_goal_duration = self.hourly_schedule[next_schedule_index]
		self.curr_goal_address = self.generate_goal_address()
		self.commuting = True
		curr_goal_dict = self.s_mem.get_knowledge(self.curr_goal_address)
		self.curr_goal_pos = np.array([curr_goal_dict["location"][0], curr_goal_dict["location"][1]])
		self.curr_goal_bbox = np.array(curr_goal_dict["bounding_box"])
		self.scratch["act_address"] = self.curr_goal_address
		self.scratch["act_description"] = self.curr_goal_description
		self.scratch["act_duration"] = self.curr_goal_duration
		curr_goal_start_time = datetime.combine(self.curr_time.date(), datetime.min.time()) + timedelta(minutes=sum(activity[1] for activity in self.hourly_schedule[:next_schedule_index]))
		self.scratch["act_start_time"] = curr_goal_start_time.strftime("%B %d, %Y, %H:%M:%S")
		self.curr_goal_end_time = curr_goal_start_time + timedelta(minutes=self.curr_goal_duration)
		print(f"{self.name} {self.curr_goal_description} end at {self.curr_goal_end_time}")
		self.logger.info(f"{self.name} planned to {self.curr_goal_description} at {self.curr_goal_address} for {self.curr_goal_duration} minutes.")
	
	def react_buy_good(self, focused_event):
		context = ""
		curr_descs = []
		for event in focused_event["events"]:
			curr_descs.append(event["event_description"])
		curr_descs = set(curr_descs)
		for curr_desc in curr_descs:
			context +=  f"{curr_desc}. "
		context += f"\nHere is a brief description of {self.name}:\n{self.get_character_description()}\n"
		focal_points = focused_event["curr_event"].event_keywords
		focal_points_embedding = []
		for focal_point in focal_points:
			if focal_point != "":
				focal_points_embedding.append(self.generator_embedding.get_embedding(focal_point, caller="embed_focal_point") if focal_point not in self.e_mem.experience_embedding.keys() else self.e_mem.experience_embedding[focal_point])
		retrieved = self.e_mem.new_retrieve(self.curr_time, focal_points, focal_points_embedding, 50)
		retrieved_strs = []
		for focal_point_retrieved in retrieved:
			for event in retrieved[focal_point_retrieved]:
				retrieved_strs.append(event["event_description"])
		retrieved_strs = list(set(retrieved_strs))
		return self.generate_decide_to_buy(focused_event, context, "\n".join(retrieved_strs))

	def generate_decide_to_buy(self, focused_event, context, retrieved_str):
		prompt_part1 = f"Context for the task:\nPART 1. {context} Here is the memory that is in {self.name}'s head: {retrieved_str}\n"
		prompt_part2 = f"Task: I am in a store. {focused_event['curr_event'].event_description}. Given the above, only return the name of the item you want to buy. If you do not want to buy anything, return 'nothing'. Do not provide any explanation, reasoning, or additional text."
		prompt = prompt_part1 + "---\n" + prompt_part2
		response = self.generator.generate(prompt, caller="generate_decide_to_buy")
		response = response.strip()
		for good in focused_event['curr_event'].event_keywords:
			if good in response:
				return good
		return None

	def react_converse(self, obs, focused_event, target_knowledge): # only support conversation currently
		def converse_end_procedure():
			target_agent_name = focused_event["curr_event"].event_keywords[0]
			if "avatar:" in target_agent_name:
				target_agent_name = target_agent_name.split("avatar:")[1]
			kws = [target_agent_name, "chat", self.name]
			desc = self.generate_conversation_summarization(self.e_mem.curr_chat)
			if desc not in self.e_mem.experience_embedding.keys():
				if desc != "":
					embedding = self.generator_embedding.get_embedding(desc, caller="embed_chat")
				else:
					embedding = None
			else:
				embedding = self.e_mem.experience_embedding[desc]
			if embedding is not None:
				self.e_mem.add_memory(event_type="chat", event_time=self.curr_time, event_position=self.pose[:3], event_place=obs['current_place'], event_keywords=kws, event_img=None, event_description=desc, event_embedding=embedding, event_poignancy=self.generate_poig_score(desc))
			if self.last_new_schedule_time is None or (self.curr_time - self.last_new_schedule_time).total_seconds() > self.generate_new_schedule_freq:
				print("conversation summary:", desc)
				try:
					self.hourly_schedule = self.generate_new_decomp_schedule(desc)
					print(self.hourly_schedule)
					self.scratch["hourly_schedule"] = self.hourly_schedule
				except Exception as e:
					print(f"Error in generating new schedule: {e}")
				self.last_new_schedule_time = self.curr_time
		if sum(1 for name, _ in self.e_mem.curr_chat if name == self.name) >= 8:
			converse_end_procedure()
			self.e_mem.curr_chat_end = False
			self.e_mem.curr_chat = []
			return False
		if self.e_mem.curr_chat_end:
			converse_end_procedure()
			self.e_mem.curr_chat_end = False
			self.e_mem.curr_chat = []
			return False
		return self.generate_decide_to_talk(obs, focused_event, target_knowledge)
	
	def generate_event_triple_from_rgb(self, rgb):
		prompt = "Task: Turn the input image into (subject, predicate, object). \nOutput examples: (Woman, eat, breakfast), (Man, brew, coffee), (Child, is, sleep) \nOutput:"
		response = self.generator.generate(prompt, img=rgb, caller="generate_event_triple_from_rgb")
		subject, predicate, object = response.strip("()").split(", ")
		return subject, predicate, object
	
	def generate_event_triple_from_text(self, text):
		prompt = f"Task: Turn the input text '{text}' into (subject, predicate, object). \nOutput examples: (Woman, eat, breakfast), (Man, brew, coffee), (Child, is, sleep) \nOutput:"
		response = self.generator.generate(prompt, caller="generate_event_triple_from_text")
		subject, predicate, object = response.strip("()").split(", ")
		return subject, predicate, object
	
	def generate_action_event_triple(self, action_description):
		with open ("agents/prompts/gen_agent/v2/generate_event_triple_v1.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", self.name)
		prompt = prompt.replace("!<INPUT 1>!", action_description)
		prompt = prompt.replace("!<INPUT 2>!", self.name)
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		response = self.generator.generate(prompt, caller="generate_action_event_triple")
		try:
			cr = response.strip()
			cr = [i.strip() for i in cr.split(")")[0].split(",")]
			assert len(cr) == 2
			cr = (self.name, cr[0], cr[1])
		except:
			cr = (self.name, "is", "idle")
		return cr
	
	def generate_to_turn(self, speech): 
		prompt = f"My name is {self.name} and I hear someone is talking: {speech} \nShould I turn around to the source of the sound? Respond with only 'yes' or 'no'."
		response = self.generator.generate(prompt, caller="generate_to_turn")
		if "yes" in response.lower() and "no" not in response.lower():
			return True
		return False

	def describe_knowledge(self, knowledge):
		if knowledge is None:
			return "No knowledge."
		desc = ""
		for key, value in knowledge.items():
			if ".png" in str(value):
				continue
			if "_idx" in key:
				continue
			desc += f"{key}: {value}\n"
		desc = f"{{\n{desc}}}\n"
		return desc

	def generate_decide_to_talk(self, obs, focused_event, target_knowledge):
		target_agent_name = focused_event["curr_event"].event_keywords[0].split('avatar:')[1]
		last_chat = self.e_mem.get_last_chat(target_agent_name)
		if last_chat: 
			last_chat_time = last_chat["time"]
			last_chat_desciption = last_chat["description"]
		else:
			last_chat_time = ""
			last_chat_desciption = ""
		context = ""
		curr_descs = []
		for event in focused_event["events"]:
			curr_descs.append(event["event_description"])
		curr_descs = set(curr_descs)
		for curr_desc in curr_descs:
			context +=  f"{curr_desc}. "
		context += f"\nHere is a brief description of {self.name}:\n{self.get_character_description()}\n"
		context += f"Knowledge about {target_agent_name}:\n{self.describe_knowledge(target_knowledge)}"
		curr_time = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
		if self.curr_goal_address == obs['current_place'] and "waiting" not in self.curr_goal_description: 
			curr_act_description = f"{self.name} is already {self.curr_goal_description}"
		elif "waiting" in self.curr_goal_description:
			curr_act_description = f"{self.name} is {self.curr_goal_description}"
		else: 
			curr_act_description = f"{self.name} is on the way to {self.curr_goal_description}"
		target_curr_act_description = ""
		for event in self.obs["events"]:
			if event["type"] == "speech" and event["content"]["from_name"] == target_agent_name:
				target_curr_act_description = f"{target_agent_name} is talking: {event['content']['utterance']}"
		with open ("agents/prompts/gen_agent/v2/decide_to_talk_v2.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", context)
		prompt = prompt.replace("!<INPUT 1>!", curr_time)
		prompt = prompt.replace("!<INPUT 2>!", self.name)
		prompt = prompt.replace("!<INPUT 3>!", target_agent_name)
		prompt = prompt.replace("!<INPUT 4>!", last_chat_time)
		prompt = prompt.replace("!<INPUT 5>!", last_chat_desciption)
		prompt = prompt.replace("!<INPUT 6>!", curr_act_description)
		prompt = prompt.replace("!<INPUT 7>!", target_curr_act_description)
		prompt = prompt.replace("!<INPUT 8>!", self.name)
		prompt = prompt.replace("!<INPUT 9>!", target_agent_name)
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		response = self.generator.generate(prompt, caller="generate_decide_to_talk")
		if "yes" in response.lower():
			self.logger.debug("{self.name}@generate_decide_to_talk:True")
			return True
		self.logger.debug("{self.name}@generate_decide_to_talk:False")
		return False
	
	def converse(self, obs, focused_event, target_agent_state=None):
		target_agent_name = focused_event["curr_event"].event_keywords[0].split('avatar:')[1]
		focal_points = [target_agent_name]
		focal_points_embedding = []
		for focal_point in focal_points:
			if focal_point != "":
				focal_points_embedding.append(self.generator_embedding.get_embedding(focal_point, caller="embed_focal_point") if focal_point not in self.e_mem.experience_embedding.keys() else self.e_mem.experience_embedding[focal_point])
		retrieved = self.e_mem.new_retrieve(self.curr_time, focal_points, focal_points_embedding, 50)
		retrieved_str = ""
		for focal_point_retrieved in retrieved:
			for event in retrieved[focal_point_retrieved]:
				retrieved_str += f"{event['event_description']}\n"
		statements = retrieved_str
		# target_agent_in_groups = False
		for group in self.scratch["groups"]:
			if target_agent_name in group["members"]:
				statements += f'{target_agent_name} is also in the {group["name"]} which is {group["description"][0].lower() + group["description"][1:]} located at the {group["place"]}.\n'
			else:
				statements += f"{self.name} and {target_agent_name} don't have group relationships"
		if statements == "":
			statements = f"{self.name} and {target_agent_name} don't know each other."
		relationship = self.generate_chat_summarize_relationship(target_agent_name, statements)
		last_chat = ""
		for chat_single in self.e_mem.curr_chat[-4:]:
			last_chat += ": ".join(chat_single) + "\n"
		if last_chat:
			focal_points = [f"{relationship}", 
                      f"{target_agent_name} is {target_agent_state}" if target_agent_state is not None else f"{target_agent_name}", 
                      last_chat]
		else:
			focal_points = [f"{relationship}", 
                      f"{target_agent_name} is {target_agent_state}" if target_agent_state is not None else f"{target_agent_name}"]
		focal_points_embedding = []
		for focal_point in focal_points:
			if focal_point != "":
				focal_points_embedding.append(self.generator_embedding.get_embedding(focal_point, caller="focal_point") if focal_point not in self.e_mem.experience_embedding.keys() else self.e_mem.experience_embedding[focal_point])
		retrieved = self.e_mem.new_retrieve(self.curr_time, focal_points, focal_points_embedding, 15)
		current_place = obs["current_place"] if obs["current_place"] is not None else "outdoor"
		utt, end = self.generate_one_utterance(target_agent_name, target_agent_state, current_place, relationship, retrieved, self.e_mem.curr_chat)
		self.e_mem.curr_chat += [[self.name, utt]]
		if end:
			self.e_mem.curr_chat_end = True
			kws = ["avatar" + ':' + target_agent_name, "chat", "avatar" + ':' + self.name]
			desc = self.generate_conversation_summarization(self.e_mem.curr_chat)
			if desc not in self.e_mem.experience_embedding.keys():
				if desc != "":
					embedding = self.generator_embedding.get_embedding(desc, caller="embed_chat")
				else:
					embedding = None
			else:
				embedding = self.e_mem.experience_embedding[desc]
			if embedding is not None:
				self.e_mem.add_memory(event_type="chat", event_time=self.curr_time, event_position=self.pose[:3], event_place=obs['current_place'], event_keywords=kws, event_img=None, event_description=desc, event_embedding=embedding, event_poignancy=self.generate_poig_score(desc))
		return utt, end
	
	def extract_first_json_dict(self, data_str):
		start_idx = data_str.find('{')
		end_idx = data_str.find('}', start_idx) + 1
		if start_idx == -1 or end_idx == 0:
			return None
		json_str = data_str[start_idx:end_idx]
		try:
			json_dict = json.loads(json_str)
			return json_dict
		except json.JSONDecodeError:
			return None
		
	def human_readable_time_difference(self, current_time, past_time):
		time_diff = current_time - past_time
		total_seconds = int(time_diff.total_seconds())
		days = total_seconds // 86400
		hours = (total_seconds % 86400) // 3600
		minutes = (total_seconds % 3600) // 60
		seconds = total_seconds % 60
		result = []
		if days > 0:
			result.append(f"{days} day{'s' if days > 1 else ''}")
		if hours > 0:
			result.append(f"{hours} hour{'s' if hours > 1 else ''}")
		if minutes > 0:
			result.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
		if seconds > 0:
			result.append(f"{seconds} second{'s' if seconds > 1 else ''}")
		return " ".join(result)

	def generate_one_utterance(self, target_agent_name, target_agent_state, current_place, relationship, retrieved, curr_chat):
		target_agent_state_ing_form_map = {"walking": "walking", "bicycle": "bicycling", "bus": "riding the bus", "null": None}
		retrieved_strs = []
		for focal_point_retrieved in retrieved:
			for event in retrieved[focal_point_retrieved]:
				retrieved_strs.append(f"{event['event_description']}")
		retrieved_strs = set(retrieved_strs)
		retrieved_str = ""
		for retrieved_str in retrieved_strs:
			retrieved_str += f"{retrieved_str}.\n"
		curr_context = (f"{self.name} " + 
              f"was {self.curr_goal_description} " + 
              f"when {self.name} " + 
              f"saw {target_agent_name}" )
		if target_agent_state != None:
			curr_context += f" in the middle of {target_agent_state_ing_form_map[target_agent_state]}.\n"
		else:
			curr_context += ".\n"
		curr_context += f"The relationship between {self.name} and {target_agent_name}: {relationship}\n"
		curr_context += (f"{self.name} " +
              f"is initiating a conversation with " +
              f"{target_agent_name}.")
		prev_conversation_insert = "\n"
		last_chat = self.e_mem.get_last_chat(target_agent_name)
		if last_chat is not None:
			if int((self.curr_time - datetime.strptime(last_chat["time"], "%B %d, %Y, %H:%M:%S")).total_seconds() / 60) > 480: 
				prev_conversation_insert = ""
			else:
				prev_conversation_insert += f"{self.human_readable_time_difference(current_time=self.curr_time, past_time=datetime.strptime(last_chat['time'], '%B %d, %Y, %H:%M:%S'))} minutes ago, {self.name} and {target_agent_name} were already {last_chat['description']}. This context takes place after that conversation."
		if prev_conversation_insert == "\n": 
			prev_conversation_insert = ""
		curr_chat_verbalize = ""
		for chat_single in curr_chat:
			curr_chat_verbalize += ": ".join(chat_single) + "\n"
		if curr_chat_verbalize == "":
			curr_chat_verbalize = "[The conversation has not started yet -- start it!]"
		intro_character_description = f"Here is a brief description of {self.name}.\n{self.get_character_description()}"
		with open ("agents/prompts/gen_agent/v3_ChatGPT/iterative_convo_v1.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", intro_character_description)
		prompt = prompt.replace("!<INPUT 1>!", self.name)
		prompt = prompt.replace("!<INPUT 2>!", retrieved_str)
		prompt = prompt.replace("!<INPUT 3>!", prev_conversation_insert)
		prompt = prompt.replace("!<INPUT 4>!", current_place)
		prompt = prompt.replace("!<INPUT 5>!", curr_context)
		prompt = prompt.replace("!<INPUT 6>!", self.name)
		prompt = prompt.replace("!<INPUT 7>!", target_agent_name)
		prompt = prompt.replace("!<INPUT 8>!", curr_chat_verbalize)
		prompt = prompt.replace("!<INPUT 9>!", self.name)
		prompt = prompt.replace("!<INPUT 10>!", target_agent_name)
		prompt = prompt.replace("!<INPUT 11>!", self.name)
		prompt = prompt.replace("!<INPUT 12>!", self.name)
		prompt = prompt.replace("!<INPUT 13>!", self.name)
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		response = self.generator.generate(prompt, caller="generate_one_utterance")
		try:
			gpt_response_dict = self.extract_first_json_dict(response)
			cleaned_dict = dict()
			cleaned = []
			for key, val in gpt_response_dict.items(): 
				cleaned += [val]
				cleaned_dict["utterance"] = cleaned[0]
				cleaned_dict["end"] = True
			if "f" in str(cleaned[1]) or "F" in str(cleaned[1]): 
				cleaned_dict["end"] = False
		except:
			cleaned_dict = dict()
			cleaned_dict["utterance"] = "..."
			cleaned_dict["end"] = False
		return cleaned_dict["utterance"], cleaned_dict["end"]

	def generate_chat_summarize_relationship(self, target_persona_name, statements):
		with open ("agents/prompts/gen_agent/v3_ChatGPT/summarize_chat_relationship_v2.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", statements)
		prompt = prompt.replace("!<INPUT 1>!", self.name)
		prompt = prompt.replace("!<INPUT 2>!", target_persona_name)
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		response = self.generator.generate(prompt, caller="generate_chat_summarize_relationship")
		return response
	
	def generate_conversation_summarization(self, curr_chat):
		conversation_str = ""
		for row in curr_chat: 
			conversation_str += f'{row[0]}: "{row[1]}"\n'
		with open ("agents/prompts/gen_agent/v3_ChatGPT/summarize_conversation_v1.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", conversation_str)
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		example_output = "conversing about eating dinner with Brian Carter in Boston Sail Loft at 18:00."
		special_instruction = "The output must continue the sentence above by filling in the <fill in> tag. Don't start with 'this is a conversation about...' Just finish the sentence but do not miss any important details (including who, when, where if applicable)."
		response = self.generator.generate(self.compose_instruction_prompts(prompt=prompt, example_output=example_output, special_instruction=special_instruction), caller="generate_conversation_summarization")
		return response
	
	def generate_goal_address(self):
		assert self.scratch["living_place"] is not None
		places_in_spatial_memory = self.s_mem.get_places()
		places_in_spatial_memory_formatted = [f'"{place}"' for place in places_in_spatial_memory]
		prompt = f"My living place is {self.scratch['living_place']}. Currently, I want to do this action: {self.curr_goal_description}, what is the most possible place for doing this action in the following list of places: [{','.join(places_in_spatial_memory_formatted)}]. Choose only one place from the list and output it as the only response. Do not provide any explanation, reasoning, or additional text."
		
		response = self.generator.generate(prompt, caller="generate_goal_address")
		cr = self.scratch['living_place']
		try:
			for place in places_in_spatial_memory:
				if place in response:
					cr = place
					break
		except:
			cr = self.scratch['living_place']
		return cr
	
	def revise_identity_and_generate_daily_plan(self):
		self.wake_up_hour = self.generate_wake_up_hour()
		focal_points = [f"{self.name}'s plan for {self.get_curr_date()}.",
						f"Important recent events for {self.name}'s life."]
		focal_points_embedding = []
		for focal_point in focal_points:
			if focal_point != "":
				focal_points_embedding.append(self.generator_embedding.get_embedding(focal_point, caller="embed_focal_point") if focal_point not in self.e_mem.experience_embedding.keys() else self.e_mem.experience_embedding[focal_point])
		retrieved = self.e_mem.new_retrieve(self.curr_time, focal_points, focal_points_embedding, 30)
		statements = "[Statements]\n"
		for focal_point_retrieved in retrieved:
			for event in retrieved[focal_point_retrieved]:
				statements += f"{event['event_time']}: {event['event_description']}\n"

		plan_prompt = statements + "\n"
		plan_prompt += f"Given the statements above, is there anything that {self.name} should remember as they plan for"
		plan_prompt += f" *{self.curr_time.strftime('%A %B %d')}*? "
		plan_prompt += f"If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement)\n\n"
		plan_prompt += f"Write the response from {self.name}'s perspective."
		plan_note = self.generator.generate(plan_prompt, caller="revise_identity")

		thought_prompt = statements + "\n"
		thought_prompt += f"Given the statements above, how might we summarize {self.name}'s feelings about their days up to now?\n\n"
		thought_prompt += f"Write the response from {self.name}'s perspective."
		thought_note = self.generator.generate(thought_prompt, caller="revise_identity")

		currently_prompt = f"{self.name}'s status from {(self.curr_time - timedelta(days=1)).strftime('%A %B %d')}:\n"
		currently_prompt += f"{self.scratch['currently']}\n\n"
		currently_prompt += f"{self.name}'s thoughts at the end of {(self.curr_time - timedelta(days=1)).strftime('%A %B %d')}:\n" 
		currently_prompt += ((plan_note if plan_note is not None else "") + (thought_note if thought_note is not None else "")).replace('\n', '') + "\n\n"
		currently_prompt += f"It is now {self.curr_time.strftime('%A %B %d')}. Given the above, write {self.name}'s status for {self.curr_time.strftime('%A %B %d')} that reflects {self.name}'s thoughts at the end of {(self.curr_time - timedelta(days=1)).strftime('%A %B %d')}. Write this in third-person talking about {self.name}."
		currently_prompt += f"If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement).\n\n"
		currently_prompt += "Follow this format below:\nStatus: <new status>"
		new_currently = self.generator.generate(currently_prompt, caller="revise_identity")

		self.scratch['currently'] = new_currently

		daily_req_prompt = self.get_character_description() + "\n"
		daily_req_prompt += f"Today is {self.curr_time.strftime('%A %B %d')}. Here is {self.name}'s plan today in broad-strokes (with the time of the day. e.g., have a lunch at 12:00 pm, watch TV from 7 to 8 pm).\n\n"
		daily_req_prompt += f"Follow this format (the list should have 4~6 items but no more):\n"
		daily_req_prompt += f"1. wake up and complete the morning routine at <time>, 2. ..."

		response = self.generator.generate(daily_req_prompt, caller="revise_identity")
		try:
			lines = response.strip().split('\n')
			cr = [line.split('. ', 1)[1].strip().lower() for line in lines]
		except:
			cr = ['wake up and complete the morning routine at 6:00 am',
				  'eat breakfast at 7:00 am',
				  'read a book from 8:00 am to 12:00 pm',
				  'have lunch at 12:00 pm',
				  'take a nap from 1:00 pm to 4:00 pm',
				  'relax and watch TV from 7:00 pm to 8:00 pm',
				  'finish night routine and sleep at 11:00 pm']
		cr = ([f"wake up and complete the morning routine at {self.wake_up_hour}:00 am"]
              + cr)
		return cr

	def generate_daily_plan(self):
		self.wake_up_hour = self.generate_wake_up_hour()
		with open ("agents/prompts/gen_agent/v2/daily_planning_v6.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", self.get_character_description())
		prompt = prompt.replace("!<INPUT 1>!", self.scratch['lifestyle'])
		prompt = prompt.replace("!<INPUT 2>!", self.get_curr_date())
		prompt = prompt.replace("!<INPUT 3>!", self.scratch['first_name'])
		prompt = prompt.replace("!<INPUT 4>!", f"{self.wake_up_hour}:00 am")
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		response = self.generator.generate(prompt, caller="generate_daily_plan")
		try:
			cr = []
			_cr = response.split(")")
			for i in _cr:
				if i[-1].isdigit():
					i = i[:-1].strip()
					if i[-1] == "." or i[-1] == ",":
						cr += [i[:-1].strip()]
		except:
			cr = ['wake up and complete the morning routine at 6:00 am',
				  'eat breakfast at 7:00 am',
				  'read a book from 8:00 am to 12:00 pm',
				  'have lunch at 12:00 pm',
				  'take a nap from 1:00 pm to 4:00 pm',
				  'relax and watch TV from 7:00 pm to 8:00 pm',
				  'finish night routine and sleep at 11:00 pm']
		cr = ([f"wake up and complete the morning routine at {self.wake_up_hour}:00 am"]
              + cr)
		return cr

	def generate_hourly_schedule(self):
		hour_str = ["00:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM",
					"05:00 AM", "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM",
					"10:00 AM", "11:00 AM", "12:00 PM", "01:00 PM", "02:00 PM",
					"03:00 PM", "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM",
					"08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"]
		n_m1_activity = []
		diversity_repeat_count = 3
		wake_up_hour = self.wake_up_hour
		for i in range(diversity_repeat_count):
			n_m1_activity_set = set(n_m1_activity)
			if len(n_m1_activity_set) < 5:
				n_m1_activity = []
				for count, curr_hour_str in enumerate(hour_str):
					if wake_up_hour > 0:
						n_m1_activity += ["sleeping"]
						wake_up_hour -= 1
					else:
						n_m1_activity += [self.generate_curr_hour_schedule(curr_hour_str, n_m1_activity, hour_str)]

		# Step 1. Compressing the hourly schedule to the following format:
		# The integer indicates the number of hours. They should add up to 24.
		# [['sleeping', 6], ['waking up and starting her morning routine', 1],
		# ['eating breakfast', 1], ['getting ready for the day', 1],
		# ['working on her painting', 2], ['taking a break', 1],
		# ['having lunch', 1], ['working on her painting', 3],
		# ['taking a break', 2], ['working on her painting', 2],
		# ['relaxing and watching TV', 1], ['going to bed', 1], ['sleeping', 2]]
		_n_m1_hourly_compressed = []
		prev = None
		prev_count = 0
		for i in n_m1_activity:
			if i != prev:
				prev_count = 1
				_n_m1_hourly_compressed += [[i, prev_count]]
				prev = i
			else:
				if _n_m1_hourly_compressed:
					_n_m1_hourly_compressed[-1][1] += 1

		n_m1_hourly_compressed = []
		for task, duration in _n_m1_hourly_compressed:
			task_cleaned = task
			if '\n' in task_cleaned:
				task_cleaned = task.split('\n')[0]
			n_m1_hourly_compressed += [[task_cleaned, duration * 60]]

		return n_m1_hourly_compressed

	def generate_curr_hour_schedule(self, curr_hour_str, p_f_ds_hourly_org, hour_str):
		with open ("agents/prompts/gen_agent/v2/generate_hourly_schedule_v2.txt", "r") as f:
			prompt_template = f.read()
		schedule_format = ""
		for i in hour_str:
			schedule_format += f"[{self.get_curr_date()} -- {i}]"
			schedule_format += f" Activity: [Fill in]\n"
		schedule_format = schedule_format[:-1]

		prior_schedule = ""
		if p_f_ds_hourly_org:
			prior_schedule = "\n"
			for count, i in enumerate(p_f_ds_hourly_org):
				prior_schedule += f" {self.get_curr_date()} -- {hour_str[count]}] Activity: {self.scratch['first_name']} is {i}\n"

		intermission_str = f"Here the originally intended hourly breakdown of"
		intermission_str += f" {self.scratch['first_name']}'s schedule today: "
		for count, i in enumerate(self.daily_plan):
			intermission_str += f"{str(count + 1)}) {i}, "
		intermission_str = intermission_str[:-2]
		prompt = prompt_template.replace("!<INPUT 0>!", schedule_format)
		prompt = prompt.replace("!<INPUT 1>!", self.get_character_description())
		prompt = prompt.replace("!<INPUT 2>!", prior_schedule)
		prompt = prompt.replace("!<INPUT 3>!", intermission_str)
		prompt = prompt.replace("!<INPUT 4>!", "")
		prompt = prompt.replace("!<INPUT 5>!", f" {self.get_curr_date()} -- {curr_hour_str}] Activity: {self.scratch['first_name']} is")
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		response = self.generator.generate(prompt, caller="generate_curr_hour_schedule")
		try:
			cr = response.strip()
			if '[' in cr:
				cr = cr.split("[")[0].strip()
			if cr[-1] == ".":
				cr = cr[:-1]
		except:
			cr = "sleeping"
		return cr

	def generate_wake_up_hour(self):
		with open ("agents/prompts/gen_agent/v2/wake_up_hour_v1.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", self.get_character_description())
		prompt = prompt.replace("!<INPUT 1>!", self.scratch['lifestyle'])
		prompt = prompt.replace("!<INPUT 2>!", self.scratch['first_name'])
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]

		response = self.generator.generate(prompt, caller="generate_wake_up_hour")
		try:
			cr = int(response.strip().lower().split("am")[0])
		except:
			cr = 8
		return cr

	def get_next_schedule_index(self):
		today_min_elapsed = 0
		today_min_elapsed += self.curr_time.hour * 60
		today_min_elapsed += self.curr_time.minute

		curr_index = 0
		elapsed = 0
		for task, duration in self.hourly_schedule:
			elapsed += duration
			if elapsed > today_min_elapsed:
				return curr_index
			curr_index += 1

		return curr_index

	def get_curr_date(self):
		if self.curr_time is None:
			return None
		return self.curr_time.strftime("%A %B %d")

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
		   World Daily Schedule: Dolores is planning to stay at home all day and
			 never go out. [missed now]
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
			Current Date: {self.get_curr_date()}
		"""
	
	def compose_instruction_prompts(self, prompt, example_output, special_instruction):
		prompt = '"""\n' + prompt + '\n"""\n'
		prompt += f"Output the response to the prompt above. {special_instruction}\n"
		prompt += "Example output:\n"
		prompt += str(example_output)
		return prompt