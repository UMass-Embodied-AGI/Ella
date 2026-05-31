import os
from collections import defaultdict
from PIL import Image
import threading
from threading import Thread

import time
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
from bisect import bisect_left, bisect_right

from .sg.builder.builder import Builder, BuilderConfig, VolumeGridBuilderConfig
from .sg.builder.object import Object, AGENT_TAGS
from tools.model_manager import global_model_manager
from vico.tools.utils import *

class SemanticMemory:
    def __init__(self, storage_path, detect_interval=-1, fov=90.0, region_layer=False, debug=False, logger=None):
        self.storage_path = storage_path
        self.detect_interval = detect_interval
        self.fov = fov
        self.debug = debug
        self.logger = logger
        self.knowledge_path = os.path.join(storage_path, "knowledge.json")
        self.knowledge = {}
        self.knowledge_feature_path = os.path.join(storage_path, "knowledge_feature.pkl")
        self.knowledge_feature = {}
        self.places = []
        self.transit_places = []
        self.agents = []
        self.scene_graph_dict = {}
        self.current_place = "open space"
        self.get_sg(self.current_place)
        self.num_frames = 0
        self.last_processed_rgb = None
        self.object_builder = None
        self.region_builder = None
        self.saving_lock = threading.Lock()

        if self.detect_interval != -1:
            from sg.builder.object import ObjectBuilder, ObjectBuilderConfig, AGENT_TAGS, VEHICLE_TAGS
            os.makedirs(os.path.join(storage_path, "object"), exist_ok=True)
            self.object_builder = ObjectBuilder(ObjectBuilderConfig(
                fov=self.fov, debug=self.debug, output_path=os.path.join(storage_path, "object"), logger=self.logger))
            if region_layer:
                from sg.builder.region import RegionBuilder
                self.region_builder = RegionBuilder(vg_builder=self.get_sg(self.current_place).volume_grid_builder,
                    obj_builder=self.object_builder, logger=self.logger,debug=self.debug, output_dir=os.path.join(storage_path, "region"))

        self.load_memory()
        self.SIMILARITY_THRESHOLD = 0.75

    def get_sg(self, place=None):
        if place is None:
            place = "open space"
        if place not in self.scene_graph_dict:
            output_path = f"{self.storage_path}/{place}"
            os.makedirs(output_path, exist_ok=True)
            if place == "open space":
                volume_grid_conf = VolumeGridBuilderConfig(voxel_size=0.1, nav_grid_size=0.5, depth_bound=30.0)
            else:
                volume_grid_conf = VolumeGridBuilderConfig(voxel_size=0.025, nav_grid_size=0.2, depth_bound=30.0)

            self.scene_graph_dict[place] = Builder(
                BuilderConfig(output_path=output_path,
                            volume_grid_conf=volume_grid_conf, fov=self.fov, debug=self.debug, logger=self.logger))
            if os.path.exists(f"{self.storage_path}/{place}/volume_grid.pkl"):
                print(f"Loading volume grid for {place}...")
                self.scene_graph_dict[place].volume_grid_builder.load(f"{self.storage_path}/{place}/volume_grid.pkl")
        if self.current_place is not None and self.current_place != place:
            self.scene_graph_dict[self.current_place].volume_grid_builder.save(f"{self.storage_path}/{self.current_place}/volume_grid.pkl")
        self.current_place = place
        return self.scene_graph_dict[place]

    def get_objects(self) -> dict[int, Object]:
        if self.object_builder is None:
            return {}
        return self.object_builder.objects

    def get_places(self):
        return self.places

    def get_transit_places(self):
        return self.transit_places

    def get_building_from_place(self, place):
        if place not in self.knowledge:
            self.logger.warning(f"Place {place} is not in knowledge.")
            return None
        return self.knowledge[place]['building']

    def get_name_from_position(self, position):
        if self.object_builder is None:
            return None
        x, y, z = position
        object_idx = self.scene_graph_dict[self.current_place].volume_grid_builder.get_label(x, y, z, 0.5)
        if object_idx < 0:
            return None
        objects = self.get_objects()
        if object_idx in objects:
            return objects[object_idx].name
        return None

    def get_position_from_name(self, name) -> list:
        knowledge = self.get_knowledge(name)
        if knowledge is None:
            return None
        if 'location' in knowledge:
            return knowledge['location']
        if 'object_idx' in knowledge:
            object_idx = knowledge['object_idx']
            objects = self.get_objects()
            if object_idx in objects:
                return objects[object_idx].get_position()
        return None

    def get_knowledge(self, name):
        return self.knowledge.get(name, None)

    def retrieve_knowledge(self, query):
        pass

    def load_memory(self):
        if os.path.exists(self.knowledge_path):
            with open(self.knowledge_path, 'r') as f:
                self.knowledge = json.load(f)
        else:
            self.knowledge = json.load(open(os.path.join(os.path.dirname(self.storage_path), 'seed_knowledge.json'), 'r'))
        for name, knowledge in self.knowledge.items():
            if 'coarse_type' in knowledge:
                self.places.append(name)
                if knowledge['coarse_type'] == 'transit':
                    self.transit_places.append(name)
            if 'age' in knowledge:
                self.agents.append(name)
        if os.path.exists(self.knowledge_feature_path):
            self.knowledge_feature = pickle.load(open(self.knowledge_feature_path, 'rb'))
        else:
            self.knowledge_feature = pickle.load(open(os.path.join(os.path.dirname(self.storage_path), 'seed_knowledge_feature.pkl'), 'rb'))
        if self.object_builder is not None:
            self.object_builder.load(os.path.join(self.storage_path, "object", "objects.pkl"))
            self.object_builder.num_frames = self.num_frames
        if self.region_builder is not None:
            self.region_builder.load(f"{self.storage_path}/region.json")

    def save_memory(self):
        def _save():
            with self.saving_lock:
                atomic_save(self.knowledge_path, json.dumps(self.knowledge, indent=2, default=json_converter))
                atomic_save(self.knowledge_feature_path, pickle.dumps(self.knowledge_feature))
                if self.object_builder is not None:
                    self.object_builder.save(os.path.join(self.storage_path, "object", "objects.pkl"))
                self.scene_graph_dict[self.current_place].volume_grid_builder.save(f"{self.storage_path}/{self.current_place}/volume_grid.pkl")
                if self.region_builder is not None:
                    self.region_builder.save(f"{self.storage_path}/region.json")
        
        Thread(target=_save, daemon=True).start()

    def update(self, obs):
        while self.saving_lock.locked():
            time.sleep(0.1)
        if obs['rgb'] is None:
            return 0
        cur_sg = self.get_sg(obs['current_place'])
        if self.object_builder is not None and "gt_seg_idxc_to_info" in obs:
            labels = self.object_builder.add_frame_with_gt_seg(obs['rgb'], obs['depth'], obs['segmentation'], obs['extrinsics'], obs['gt_seg_idxc_to_info'])
            num_new_objects = len(self.object_builder.new_objects)
        elif self.detect_interval > 0 and self.num_frames % self.detect_interval == 0 and (self.last_processed_rgb is None or not np.allclose(obs['rgb'], self.last_processed_rgb)):
            self.last_processed_rgb = obs['rgb']
            labels = self.object_builder.add_frame(obs['rgb'], obs['depth'], obs['extrinsics'])
            if self.debug:
                label_color = np.random.rand(max(0, int(labels.max())) + 101, 3)
                label_color[0] = 0
                label_image = label_color[labels + 100]
                Image.fromarray((label_image * 255).astype(np.uint8)).save(
                    f"{self.storage_path}/object/debug/{self.object_builder.num_frames - 1:06d}_label.png")
            num_new_objects = len(self.object_builder.new_objects)
        else:
            labels = -np.ones_like(obs['depth'], dtype=np.int32)
            num_new_objects = 0
        cur_sg.add_frame(obs['rgb'], obs['depth'], labels, obs['extrinsics'])
        self.num_frames += 1
        if num_new_objects > 0:
            # update region cluster
            if self.region_builder is not None:
                self.region_builder.add_frame()
            # match names
            if self.object_builder is None:
                new_objects = []
            else:
                new_objects = self.object_builder.get_new_objects()
            for obj in new_objects:
                if not "gt_seg_idxc_to_info" in obs:
                    matched_name = None
                    if obj.tag in AGENT_TAGS:
                        visual_sim = {}
                        # spatial_sim = []
                        for name, ft in self.knowledge_feature.items():
                            visual_sim[name] = np.dot(obj.image_ft, ft)
                        self.logger.debug(f"{obj.tag}({obj.idx}) visual similarity with names: {visual_sim}")
                        # sim = np.array(visual_sim) * 0.5 + np.array(spatial_sim) * 0.5
                        matched_name, matched_value = next(iter(top_highest_x_values(visual_sim, 1).items()))
                        if matched_value > self.SIMILARITY_THRESHOLD:
                            self.logger.debug(f"Matched new object {obj.tag}({obj.idx}) with name {matched_name}.")
                        else:
                            matched_name = None
                    if matched_name is None:
                        self.logger.debug(f"New object {obj.tag}({obj.idx}) is unmatched. A new knowledge is created")
                        matched_name = f"{obj.tag}_{obj.idx}"
                        self.knowledge[matched_name] = {}
                    obj.name = matched_name
                self.update_with_new_knowledge({obj.name: {'object_idx': obj.idx,
                                                        'detected_appearance_path': obj.appearance_path}
                                                })

        if self.debug:
            cur_sg.volume_grid_builder.get_occ_map(obs["pose"][:3], os.path.join(self.storage_path, self.current_place, f"occ_map_{cur_sg.num_frames:06d}.png"))
        if self.num_frames % 10 == 0:
            self.save_memory()
        return num_new_objects

    def update_with_new_knowledge(self, knowledge_items: dict[str, dict]):
        for name, knowledge in knowledge_items.items():
            if name not in self.knowledge:
                self.knowledge[name] = {}
            for key, value in knowledge.items():
                if key == 'location':
                    if not isinstance(value, list):
                        continue
                    if not isinstance(value[0], float):
                        continue
                self.knowledge[name][key] = value
        self.save_memory()


    def update_ga(self, obs):
        while self.saving_lock.locked():
            time.sleep(0.1)
        if obs['rgb'] is None:
            return {}
        cur_sg = self.get_sg(obs['current_place'])
        cur_objects_info = []
        labels = None
        if self.detect_interval > 0:
            self.last_processed_rgb = obs['rgb']
            labels, cur_objects = self.object_builder.add_frame_for_cur_objects(obs['rgb'], obs['depth'], obs['extrinsics'])
            for obj in cur_objects:
                matched_name = None
                if obj.tag in AGENT_TAGS:
                    visual_sim = {}
                    # spatial_sim = []
                    for name, ft in self.knowledge_feature.items():
                        visual_sim[name] = np.dot(obj.image_ft, ft)
                    self.logger.debug(f"{obj.tag}({obj.idx}) visual similarity with names: {visual_sim}")
                    # sim = np.array(visual_sim) * 0.5 + np.array(spatial_sim) * 0.5
                    matched_name, matched_value = next(iter(top_highest_x_values(visual_sim, 1).items()))
                    if matched_value > self.SIMILARITY_THRESHOLD:
                        self.logger.debug(f"Matched new object {obj.tag}({obj.idx}) with name {matched_name}.")
                    else:
                        matched_name = None
                if matched_name is None:
                    self.logger.debug(f"New object {obj.tag}({obj.idx}) is unmatched. A new knowledge is created")
                    matched_name = f"{obj.tag}_{obj.idx}"
                    self.knowledge[matched_name] = {}
                obj.name = matched_name
                self.knowledge[matched_name]['object_idx']= obj.idx
                self.knowledge[matched_name]['detected_appearance_path'] = obj.appearance_path
            for obj in cur_objects:
                obj_info = {
                    "name": obj.name,
                    "tag": obj.tag,
                    "idx": obj.idx,
                    "position": obj.get_position()
                }
                cur_objects_info.append(obj_info)
        cur_sg.add_frame(obs['rgb'], obs['depth'], labels, obs['extrinsics'])
        self.num_frames += 1
        self.save_memory()
        return cur_objects_info

    def update_transit_system_info(self, transit_system_info):
        for bus_stop_info in transit_system_info["bus"][0]["schedule"]:
            bus_stop = bus_stop_info["stop_name"]
            location = bus_stop_info["stop_pos"]
            rotation = bus_stop_info["stop_rot"]
            self.knowledge[bus_stop] = {
                "building": "open space",
                "coarse_type": "transit",
                "fine_type": "bus stop",
                "location": location,
                "rotation": rotation,
                "bounding_box": None,
            }
            self.places.append(bus_stop)
            self.transit_places.append(bus_stop)
        for bike_station in transit_system_info["bicycle"]["stations"]:
            self.knowledge[bike_station] = {
                "building": "open space",
                "coarse_type": "transit",
                "fine_type": "bike station",
                "location": transit_system_info["bicycle"]["stations"][bike_station],
                "bounding_box": None,
            }
            self.places.append(bike_station)
            self.transit_places.append(bike_station)

class EventInstance:
    def __init__(self, event_id, event_type, event_time, event_last_access_time, event_position, event_place, event_keywords, event_img, event_description, event_embedding, event_poignancy, event_expiration):
        self.event_id = event_id
        self.event_type = event_type
        self.event_time = event_time
        self.event_last_access_time = event_last_access_time
        self.event_position = event_position
        self.event_place = event_place
        self.event_keywords = event_keywords
        self.event_img = event_img
        self.event_description = event_description
        self.event_embedding = event_embedding
        self.event_poignancy = event_poignancy
        self.event_expiration = event_expiration


    def __str__(self):
        return f"EventInstance: {self.tojson()}"

    def __repr__(self):
        return self.__str__()

    def tojson(self):
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_time": self.event_time.strftime("%B %d, %Y, %H:%M:%S"),
            "event_last_access_time": self.event_last_access_time.strftime("%B %d, %Y, %H:%M:%S"),
            "event_position": self.event_position,
            "event_place": self.event_place,
            "event_keywords": self.event_keywords,
            "event_img": self.event_img,
            "event_description": self.event_description,
            # "event_embedding": self.event_embedding,
            "event_poignancy": self.event_poignancy,
            "event_expiration": self.event_expiration.strftime("%B %d, %Y, %H:%M:%S") if self.event_expiration is not None else None,
        }

    @classmethod
    def from_json(cls, json_data):
        event_time = datetime.strptime(json_data["event_time"], "%B %d, %Y, %H:%M:%S")
        event_last_access_time = datetime.strptime(json_data["event_last_access_time"], "%B %d, %Y, %H:%M:%S")
        event_expiration = datetime.strptime(json_data["event_expiration"], "%B %d, %Y, %H:%M:%S") if json_data["event_expiration"] is not None else None
        return cls(
            event_id=json_data["event_id"],
            event_type=json_data["event_type"],
            event_time=event_time,
            event_last_access_time=event_last_access_time,
            event_position=json_data["event_position"],
            event_place=json_data["event_place"],
            event_keywords=json_data["event_keywords"],
            event_img=json_data["event_img"],
            event_description=json_data["event_description"],
            event_embedding=json_data["event_embedding"],
            event_poignancy=json_data["event_poignancy"],
            event_expiration=event_expiration
        )

class EpisodicMemory:
    def __init__(self, storage_path, lm_source, debug=False, logger=None):
        self.storage_path = storage_path
        self.debug = debug
        self.logger = logger
        os.makedirs(storage_path, exist_ok=True)
        self.experience_path = os.path.join(storage_path, "experience.json")
        self.experience_embedding_path = os.path.join(storage_path, "experience_embedding.pkl")
        self.experience = []
        self.embedding_generator = global_model_manager.get_generator(lm_source, 'text-embedding-3-small', logger=self.logger)
        self.curr_chat = []
        self.curr_chat_end = False
        self.index_time = []
        self.keyword2index = defaultdict(list)
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.experience_embedding_path):
            with open(self.experience_embedding_path, 'rb') as f:
                self.experience_embedding = pickle.load(f)
        else:
            self.experience_embedding = {}
        if os.path.exists(self.experience_path):
            with open(self.experience_path, 'r') as f:
                experience_dict = json.load(f)
                for i, event_json in enumerate(experience_dict):
                    event_json["event_embedding"] = self.get_embedding(event_json["event_description"])
                    event = EventInstance.from_json(event_json)
                    self.experience.append(event)
                    self.index_time.append(event.event_time)
                    for keyword in event.event_keywords:
                        self.keyword2index[keyword].append(i)

    def save_memory(self):
        experience_tojson = []
        for event in self.experience:
            if event.event_description not in self.experience_embedding:
                self.experience_embedding[event.event_description] = event.event_embedding
            event_tojson = event.tojson()
            event_tojson.pop('event_embedding', None)
            experience_tojson.append(event_tojson)
        atomic_save(self.experience_path, json.dumps(experience_tojson, indent=2, default=json_converter))
        atomic_save(self.experience_embedding_path, pickle.dumps(self.experience_embedding))

    def save_memory_incremental(self, this_experience):
        # Support incremental memory saving later if large file writing speed is still a problem with pickle dump, mind that remove_memory() and clear_memory() also call save_memory()
        this_experience_tojson = this_experience.tojson()
        this_experience_event_embedding = this_experience_tojson['event_embedding']
        this_experience_tojson.pop('event_embedding', None)
        with open("experience.jsonl", "a") as f:
            f.write(json.dumps(this_experience_tojson) + "\n")
        with open(self.experience_embedding_path, 'ab') as f:
            pickle.dump(self.experience_embedding, f) # Mind that reading this file will require a loop to read chunks of pickled data

    # def add_chat(self, curr_time, target_agent_name, description):
    # 	if target_agent_name.lower() not in self.chat_history:
    # 		self.chat_history[target_agent_name.lower()] = []
    # 	self.chat_history[target_agent_name.lower()].append({"time": curr_time.strftime("%B %d, %Y, %H:%M:%S"), "description": description})
    # 	with open(self.chat_history_path, 'w') as f:
    # 		json.dump(self.chat_history, f)

    def get_last_chat(self, target_agent_name):
        for i in range(len(self.experience) - 1, -1, -1):
            if self.experience[i].event_type == "chat":
                # print("debug: get_last_chat experience:", self.experience[i].event_keywords[0].lower(), target_agent_name.lower())
                # print("debug: get_last_chat name:", target_agent_name.lower())
                if target_agent_name.lower() in self.experience[i].event_keywords[0].lower():
                    return {"time": self.experience[i].event_time.strftime("%B %d, %Y, %H:%M:%S"), "description": self.experience[i].event_description}
        return None
    
    def get_all_chat(self, target_agent_name, max):
        chat_descriptions = []
        for i in range(len(self.experience) - 1, -1, -1):
            if self.experience[i].event_type == "chat":
                if target_agent_name.lower() in self.experience[i].event_keywords[0].lower():
                    chat_descriptions.append(self.experience[i].event_description)
        if len(chat_descriptions) == 0:
            return None
        chat_history = []
        for chat_description in chat_descriptions:
            chat_history.append({"time": self.experience[i].event_time.strftime("%B %d, %Y, %H:%M:%S"), "description": chat_description})
        return chat_history[-max:]

    def get_embedding(self, text):
        if text not in self.experience_embedding.keys():
            self.experience_embedding[text] = self.embedding_generator.get_embedding(text)
        return self.experience_embedding[text]

    def add_memory(self, event_type, event_time, event_position, event_place, event_keywords, event_img, event_description, event_embedding, event_poignancy=None, event_expiration=None):
        event_id = f"node_{str(len(self.experience))}"
        if event_poignancy == None: event_poignancy = 4
        if event_embedding is None:
            event_embedding = self.get_embedding(event_description)
        this_experience = EventInstance(event_id, event_type, event_time, event_time, event_position, event_place, event_keywords, event_img, event_description, event_embedding, event_poignancy, event_expiration)
        self.logger.debug(f"New event added: {this_experience}")
        self.experience.append(this_experience)
        self.index_time.append(event_time)
        for keyword in event_keywords:
            self.keyword2index[keyword].append(len(self.experience) - 1)
        self.save_memory()
        return this_experience

    def get_memory(self, index):
        try:
            return self.experience[index]
        except IndexError:
            return None
        
    def cos_sim(self, a, b):
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    def extract_importance(self, events):
        importance_out = dict()
        for i, event in enumerate(events): 
            importance_out[event.event_id] = event.event_poignancy

        return importance_out

    def extract_recency(self, events):
        recency_decay = 0.995
        recency_vals = [recency_decay ** i for i in range(1, len(events) + 1)]
        recency_out = dict()
        for i, event in enumerate(events):
            recency_out[event.event_id] = recency_vals[i]
        return recency_out
    
    def extract_relevance(self, events, focal_point_embedding):
        relevance_out = dict()
        for i, event in enumerate(events):
            event_embedding = event.event_embedding
            if event_embedding is not None:
                # print(event_embedding)
                # print('-' * 50)
                # print(focal_point_embedding)
                relevance_out[event.event_id] = self.cos_sim(event_embedding, focal_point_embedding)
        # print("debug:extract_relevance:relevance_out", relevance_out)
        return relevance_out
        
    def new_retrieve(self, curr_time, focal_points, focal_points_embedding, num_events):
        # num_events for each focal point in focal_points
        recency_w = 1
        relevance_w = 1
        importance_w = 1
        retrieved = dict()
        for focal_point_index, focal_point in enumerate(focal_points):
            events = [[event.event_last_access_time, event] for event in self.experience]
            events = sorted(events, key=lambda x: x[0])
            events = [event for _, event in events]

            # print("debug:new_retrieve:events:", events)

            recency_out = self.extract_recency(events)
            recency_out = min_max_normalize_dict(recency_out)
            importance_out = self.extract_importance(events)
            importance_out = min_max_normalize_dict(importance_out)
            relevance_out = self.extract_relevance(events, focal_points_embedding[focal_point_index])
            relevance_out = min_max_normalize_dict(relevance_out)

            # print("recency_out:", recency_out)
            # print("relevance_out2:", relevance_out)

            gw = [0.5, 3, 2]
            master_out = dict()
            for key in recency_out.keys(): 
                master_out[key] = (recency_w*recency_out[key]*gw[0] 
                                + relevance_w*relevance_out[key]*gw[1] if key in relevance_out else 0 # for handling some missing embeddings
                                + importance_w*importance_out[key]*gw[2])
                
            master_out = top_highest_x_values(master_out, num_events)
            master_events_indexes = [int(key.split("node_")[1]) for key in list(master_out.keys())]
            master_events = [self.experience[index].tojson() for index in master_events_indexes]

            for index in master_events_indexes:
                self.experience[index].event_last_access_time = curr_time
            retrieved[focal_point] = master_events
        
        return retrieved

    def retrieve(self, query, curr_time, pos: list[float], k: int):
        recency_w = 0.1
        text_relevance_w = 0.5
        img_relevance_w = 0.1
        proximity_w = 0.3

        events = [[event.event_last_access_time, event] for event in self.experience]
        events = sorted(events, key=lambda x: x[0])
        events = [event for _, event in events]

        recency_out = self.extract_recency(events)
        recency_out = min_max_normalize_dict(recency_out)
        text_relevance_out = self.extract_relevance(events, self.get_embedding(query))
        text_relevance_out = min_max_normalize_dict(text_relevance_out)
        proximity_out = dict()
        for i, event in enumerate(events):
            proximity_out[event.event_id] = 1 / (np.linalg.norm(np.array(event.event_position) - np.array(pos)) + 1e-6)
        proximity_out = min_max_normalize_dict(proximity_out)

        master_out = dict()
        for key in recency_out.keys():
            master_out[key] = (recency_w * recency_out[key]
                            + text_relevance_w * text_relevance_out[key]
                            + proximity_w * proximity_out[key])

        master_out = top_highest_x_values(master_out, k)
        master_events_indexes = [int(key.split("node_")[1]) for key in list(master_out.keys())]
        master_events = [self.experience[index].tojson() for index in master_events_indexes]

        for index in master_events_indexes:
            self.experience[index].event_last_access_time = curr_time

        return master_events
        
    def retrieve_events_thoughts_by_keywords(self, events): # We use this for retrieving from perceived events
        # We do not use the same implementation used in GA because event.description can be same for different observation ids
        retrieved = []
        for event in events:
            retrieved_dict = dict()
            retrieved_dict["curr_event"] = event
            relevant_events = []
            for event_keyword in event.event_keywords:
                if event_keyword != "":
                    relevant_events.extend(self.retrieve_memory_by_keyword(event_keyword))
            retrieved_dict["events"] = list(relevant_events)
            retrieved.append(retrieved_dict)
        return retrieved

    def retrieve_latest_memory(self) -> list[dict]:
        if not self.experience:
            self.logger.warning("No event found in memory.")
            return []
        latest_time = self.experience[-1].event_time
        retrieved_experience = []
        for experience in reversed(self.experience):
            if experience.event_time == latest_time:
                retrieved_experience.append(experience.tojson())
            else:
                break
        return retrieved_experience

    def retrieve_memory_by_time(self, time):
        index = bisect_left(self.index_time, time)
        retrieved_experience = []
        for i in range(index, len(self.experience)):
            if self.experience[i].event_time == time:
                retrieved_experience.append(self.experience[i].tojson())
            else:
                break
        if retrieved_experience:
            return retrieved_experience
        self.logger.warning(f"No event found at exact time {time}, return the closest event.")
        if index < len(self.experience):
            return self.experience[index].tojson()

    def retrieve_memory_by_place(self, place):
        retrieved_experience = []
        for event in self.experience:
            if event.event_place == place:
                retrieved_experience.append(event.tojson())
        if retrieved_experience:
            return retrieved_experience
        self.logger.warning(f"No event found at place {place}.")
        return None

    def retrieve_memory_by_keyword(self, keyword):
        if keyword in self.keyword2index:
            retrieved_experience = []
            for index in self.keyword2index[keyword]:
                retrieved_experience.append(self.experience[index].tojson())
            return retrieved_experience
        self.logger.debug(f"No event found with keyword {keyword}.")
        return None

    def update_memory_last_action(self):
        if not self.experience:
            return
        latest_time = self.experience[-1].event_time
        for experience in reversed(self.experience):
            if experience.event_time != latest_time:
                break
            if experience.event_type == "action":
                experience.event_description += " But failed."
                return

    def remove_memory(self, index):
        if index < len(self.experience):
            del self.experience[index]
            self.save_memory()

    def clear_memory(self):
        self.experience.clear()
        self.save_memory()

    def __len__(self):
        return len(self.experience)

    def __getitem__(self, index):
        return self.experience[index]

    def __iter__(self):
        return iter(self.experience)

    def __str__(self):
        return f"EpisodicMemory: {self.experience}"

    def __repr__(self):
        return self.__str__()