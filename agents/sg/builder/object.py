import logging
from collections import deque
import numpy as np
from PIL import Image, ImageOps
from dataclasses import dataclass, field
import time
import pickle
import os
import sys
from logging import Logger
from .volume_grid import VolumeGridBuilder, VolumeGridBuilderConfig
current_directory = os.getcwd()
sys.path.insert(0, current_directory)
from vico.tools.utils import atomic_save

@dataclass
class ObjectBuilderConfig:
    device: str = 'cuda'
    threshold: float = 0.7
    merge_interval: int = 20
    depth_bound: float = 30.0
    building_voxel_size: float = 0.1
    object_voxel_size: float = 0.025 # for small objects
    object_min_detection: tuple = (3, 5) # at least 3 detections among 5 frames
    detection_min_pixel_ratio: float = 0.001
    dynamic_object_merge_threshold: float = 0.7
    denoise_param: dict = field(default_factory=lambda: {"type": "radius", "radius_factor": 5, "min_points": 9})
    debug: bool = False
    output_path: str = "output"
    logger: Logger = None

@dataclass
class Appearance:
    rgb: np.ndarray
    frame_num: int

AGENT_TAGS = ["person", "businessman", "man", "girl", "boy", "avatar", "couple", "soldier", "doctor", "police", "firefighter", "worker"]
VEHICLE_TAGS = ["vehicle", "car", "truck", "bus", "motorcycle", "bicycle", "bus"]
DYNAMIC_OBJECTS = AGENT_TAGS + VEHICLE_TAGS
BUILDING_TAGS = ["building", "house", "apartment", "skyscraper", "office", "factory", "warehouse", "school", "hospital", "church", "temple", "mosque", "stadium", "bridge", "tunnel"]

class Object:
    def __init__(self, idx: int, points: np.ndarray, colors: np.ndarray,
                    appearance: np.ndarray, tag: str, image_ft: np.ndarray,
                    conf: ObjectBuilderConfig, frame_num: int, name: str = None):
        # idx is 0-based, while -1 means background, -100 means Unknown
        self.idx = idx
        self.voxel_size = conf.building_voxel_size if tag in BUILDING_TAGS else conf.object_voxel_size
        self.volume_grid_builder = VolumeGridBuilder(VolumeGridBuilderConfig(voxel_size=self.voxel_size, depth_bound=conf.depth_bound))
        self.volume_grid_builder.add_points(points, colors, np.ones(points.shape[0], dtype=np.int32))
        self.appearance_list = deque(maxlen=5)
        self.appearance_list.append(Appearance(appearance, frame_num))
        self.tag = tag
        self.image_ft = image_ft
        self.detect_num = 1
        self.created_frame = frame_num
        self.last_detected_frame = frame_num
        self.name = name
        self.not_saved = True
    
    def denoise(self, param: dict) -> int:
        # denoise the object and return the number of points left
        if param["type"] == "radius":
            self.volume_grid_builder.radius_denoise(param["min_points"], param["radius_factor"] * self.voxel_size)
            return self.volume_grid_builder.get_size()
        else:
            raise NotImplementedError

    def get_bound(self) -> tuple[np.ndarray, np.ndarray]:
        return self.volume_grid_builder.get_bound()

    def get_position(self) -> list[float]:
        bound = self.get_bound()
        pos = (bound[0] + bound[1]) / 2
        return pos.tolist()
    
    def add_frame(self, rgb: np.ndarray, depth: np.ndarray, fov: float, camera_ext: np.ndarray, mask: np.ndarray, image_ft: np.ndarray, frame_num: int, appearance: np.ndarray):
        if self.tag in DYNAMIC_OBJECTS: # need to clear the volume grid
            self.volume_grid_builder.clear()
        label = np.where(mask, 0, -100)
        self.volume_grid_builder.add_frame(rgb, depth, label.astype(np.int32), fov, camera_ext)
        if self.image_ft is not None:
            self.image_ft = (self.image_ft * self.detect_num + image_ft) / (self.detect_num + 1)
        self.appearance_list.append(Appearance(appearance, frame_num))
        self.last_detected_frame = frame_num
        self.detect_num += 1
        self.not_saved = True

    
    def add_points(self, points: np.ndarray, colors: np.ndarray):
        self.volume_grid_builder.add_points(points, colors, np.ones(points.shape[0], dtype=np.int32))
        self.not_saved = True
    
    def get_points(self):
        points, colors, labels = self.volume_grid_builder.get_points()
        return points, colors
    
    def get_overlap(self, other: "Object"):
        return self.volume_grid_builder.get_overlap(other.volume_grid_builder)
    
    def visualize(self):
        self.volume_grid_builder.visualize()

    def save(self, path: str):
        if not self.not_saved:
            return
        if not os.path.exists(path):
            os.makedirs(path)
        for i, appearance in enumerate(self.appearance_list):
            atomic_save(os.path.join(path, f"appearance_{i}.png"), Image.fromarray(appearance.rgb))

        atomic_save(os.path.join(path, "tag.txt"), self.tag)
        self.volume_grid_builder.save(os.path.join(path, "volume_grid.pkl"))
        self.not_saved = False

    def load(self, path: str):
        conf = self.volume_grid_builder.conf
        self.volume_grid_builder.vg_backend = None
        self.volume_grid_builder = VolumeGridBuilder(conf)
        self.volume_grid_builder.load(os.path.join(path, "volume_grid.pkl"))

    def __str__(self) -> str:
        return f"{self.tag if self.name is None else self.name} ({self.idx})"

class ObjectBuilder:
    def __init__(self, conf: ObjectBuilderConfig):
        from .model import RAMWrapper, DINOWrapper, SAMWrapper, CLIPWrapper
        from tools.model_manager import global_model_manager
        self.conf = conf
        self.debug = conf.debug
        self.output_path = conf.output_path
        os.makedirs(os.path.join(self.output_path, "debug"), exist_ok=True)
        self.logger = conf.logger
        self.ram: RAMWrapper = global_model_manager.get_model("ram")
        self.dino: DINOWrapper = global_model_manager.get_model("dino")
        self.sam: SAMWrapper = global_model_manager.get_model("sam")
        self.clip: CLIPWrapper = global_model_manager.get_model("clip")
        self.objects: dict[int, Object] = {}
        self.new_objects: list[int] = []
        self.curr_objects: list[int] = []
        self.num_frames = 0
    
    @staticmethod
    def _crop_image(rgb: np.ndarray, box: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        if mask is not None:
            rgb = rgb.copy()
            rgb[~mask] = 0
        x1, y1, x2, y2 = box
        cropped_rgb = rgb[y1:y2, x1:x2]

        # need to pad the image to square for CLIP
        cropped_image = Image.fromarray(cropped_rgb)
        width, height = cropped_image.size
        max_dim = max(width, height)
        padding = (
            (max_dim - width) // 2,
            (max_dim - height) // 2,
            (max_dim - width + 1) // 2,
            (max_dim - height + 1) // 2,
        )
        padded_image = ImageOps.expand(cropped_image, padding)
        return np.array(padded_image)
    
    @staticmethod
    def _iou(min1: np.ndarray, max1: np.ndarray, min2: np.ndarray, max2: np.ndarray):
        volume1 = np.maximum(0, max1 - min1).prod()
        volume2 = np.maximum(0, max2 - min2).prod()
        min_all = np.maximum(min1, min2)
        max_all = np.minimum(max1, max2)
        inter = np.maximum(0, max_all - min_all).prod()
        return inter / (volume1 + volume2 - inter)

    def _ban_tag(self, tag: str):
        remove_classes = [
            "room", "kitchen", "office", "house", "home", "corner", "road", "city street", "snow", "snowy",
            "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", "man", "woman", "skateboarder", "wind",
            "apartment", "image", "city", "blue", "skylight", "hallway", 'urban', "city view", "sky", "ledge", "crack",
            "bureau", "modern", "salon", "doorway", "wall lamp", "wood floor", "wall", "floor", "ceiling", "sea", "hat",
            "game", "screenshot", "video game", "view", "road", "sky", "urban", "city street", "alley", "pole", "wing"
            "tie", "jeans", "suit", "business suit", "dress", "wear", "stand", "cocktail dress", "black", "walk", "car", "marble"
        ]
        color_names = [
            "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
            "black", "white", "gray", "silver", "gold", "beige", "cyan", "magenta",
            "turquoise", "lavender", "violet", "maroon", "navy", "indigo", "teal",
            "olive", "coral", "salmon", "tan", "ivory", "khaki", "ruby", "emerald",
            "sapphire", "amber", "peach", "plum", "azure", "mint", "lime", "crimson",
            "charcoal", "bronze", "cream", "lilac", "fuchsia", "rose", "mustard",
            "slate", "periwinkle", "turmeric", "cobalt", "sepia", "cerulean", "aqua",
            "mauve", "rust", "champagne", "eggplant", "jade", "onyx", "pearl",
            "tangerine", "vermillion", "wheat", "zaffre", "chartreuse", "mahogany",
            "cinnamon", "denim", "ecru", "garnet", "honeydew", "isabelline", "jet",
            "keppel", "linen", "malachite", "orchid", "puce", "quartz", "russet",
            "sienna", "topaz", "umber", "vanilla", "wisteria", "xanadu", "yellowgreen",
            "zinnwaldite", "dark"
        ]
        remove_classes.extend(color_names)
        if tag in remove_classes:
            return True
        for remove_class in remove_classes:
            if remove_class in [t.lower() for t in tag.split()]:
                return True
        return False
    
    def add_frame(self, rgb: np.ndarray, depth: np.ndarray, fov: float, camera_ext: np.ndarray):
        self.new_objects.clear()
        self.curr_objects.clear()

        labels = -np.ones_like(depth, dtype=np.int32) # -1 - background -100 - Unknown
        # Step 1: Detect objects
        start = time.time()
        tags = self.ram.predict(rgb)
        self.logger.debug(f"RAM time: {start}, {time.time()}")
        tags = [tag for tag in tags if not self._ban_tag(tag)]
        tags.extend(["building", "person"])
        tags = list(set(tags))
        if self.debug:
            with open(os.path.join(self.output_path, "debug", f"{self.num_frames:06d}_tag.txt"), "w") as f:
                f.write(str(tags))
            boxes, box_tags, annotate = self.dino.predict(rgb, tags, annotate=True)
            Image.fromarray(annotate).save(os.path.join(self.output_path, "debug", f"{self.num_frames:06d}_dino.png"))
            if boxes.shape[0] > 0:
                masks, annotate = self.sam.predict(rgb, boxes, annotate=True)
                Image.fromarray(annotate.transpose(1, 0, 2, 3).reshape(rgb.shape[0], -1, 3)).save(os.path.join(self.output_path, "debug", f"{self.num_frames:06d}_sam.png"))
        else:
            start = time.time()
            boxes, box_tags = self.dino.predict(rgb, tags)
            self.logger.debug(f"DINO time: {start}, {time.time()}")
            if boxes.shape[0] > 0:
                start = time.time()
                masks = self.sam.predict(rgb, boxes)
                self.logger.debug(f"SAM time: {start}, {time.time()}")

        if boxes.shape[0] > 0:
            # ensure masks are not overlapped
            # acc_mask = np.zeros_like(masks[0])
            # for i in range(boxes.shape[0]):
            #     masks[i] &= (depth < self.conf.depth_bound)
            #     masks[i] &= ~acc_mask
            #     acc_mask |= masks[i]
            
            appearances = [ObjectBuilder._crop_image(rgb, box, mask) for box, mask in zip(boxes, masks)]
            start = time.time()
            image_fts = self.clip.predict_image(appearances)
            self.logger.debug(f"CLIP time: {start}, {time.time()}")

            start = time.time()
            areas = []
            cur_objects: list[Object] = []
            for box, tag, mask, image_ft, appearance in zip(boxes, box_tags, masks, image_fts, appearances):
                mask = mask & (depth < self.conf.depth_bound)
                if self._ban_tag(tag) or mask.sum() / mask.size < self.conf.detection_min_pixel_ratio:
                    areas.append(0)
                    cur_objects.append(None)
                    continue
                label = np.where(mask, 0, -100)
                points, colors, _ = VolumeGridBuilder._img_to_pcd(rgb, depth, label, fov, camera_ext)
                obj = Object(0, points, colors, appearance, tag, image_ft, self.conf, self.num_frames)
                if obj.denoise(self.conf.denoise_param) < 5:
                    areas.append(0)
                    cur_objects.append(None)
                    continue

                areas.append(mask.sum())
                min_bound, max_bound = obj.get_bound()
                self.logger.debug(f"Detect object: {tag}, {min_bound}, {max_bound}")
                cur_objects.append(obj)

            # Step 1.5: Remove seriously overlapped objects
            for i, cur_obj in enumerate(cur_objects):
                for j, obj in enumerate(cur_objects):
                    if i == j or cur_obj is None or obj is None:
                        continue
                    if cur_obj.get_overlap(obj) > 0.8 and areas[i] < areas[j]:
                        cur_objects[i] = None
                        break

            # Step 2: Merge current objects with existing objects
            for cur_obj, appearance, mask in zip(cur_objects, appearances, masks):
                if cur_obj is None:
                    continue

                # distinguish building(since it's large) and other objects
                if cur_obj.tag in BUILDING_TAGS:
                    objects = [obj for obj in self.objects.values() if obj.tag in BUILDING_TAGS]
                else:
                    objects = [obj for obj in self.objects.values() if not obj.tag in BUILDING_TAGS]

                if len(objects) == 0:
                    self._new_object(cur_obj)
                    self.curr_objects.append(cur_obj.idx)
                    labels[mask] = cur_obj.idx
                    continue
                
                sim = self._compute_sim(cur_obj, objects)
                merge_obj_idx = sim.argmax()

                if sim[merge_obj_idx] > self.conf.threshold:
                    self.logger.debug(f"Merge object: {cur_obj} into {objects[merge_obj_idx]}")

                    objects[merge_obj_idx].add_frame(rgb, depth, fov, camera_ext, mask, cur_obj.image_ft, self.num_frames, appearance)
                    objects[merge_obj_idx].denoise(self.conf.denoise_param)
                    labels[mask] = objects[merge_obj_idx].idx
                    self.curr_objects.append(objects[merge_obj_idx].idx)
                    del cur_obj
                else:
                    self._new_object(cur_obj)
                    self.curr_objects.append(cur_obj.idx)
                    labels[mask] = cur_obj.idx
            
            self.logger.debug(f"Object process time: {start}, {time.time()}")
        
        self.num_frames += 1

        # Step 3: Filter objects
        # objects = []
        # for obj in self.objects:
        #     if obj.detect_num < self.conf.object_min_detection[0] and \
        #        self.num_frames - obj.created_frame >= self.conf.object_min_detection[1]:
        #         continue
        #     objects.append(obj)
        # self.objects = objects
        
        if self.num_frames % self.conf.merge_interval == 0:
            all_objects = list(self.objects.values())
            for obj in all_objects:
                if len(self.objects) <= 1:
                    break
                others = [o for o in self.objects.values() if o.idx != obj.idx]
                sim = self._compute_sim(obj, others)
                merge_obj_idx = sim.argmax()
                if sim[merge_obj_idx] > self.conf.threshold:
                    self.logger.debug(f"Post merge object: {obj} into {others[merge_obj_idx]}")
                    others[merge_obj_idx].add_points(*obj.get_points())
                    del self.objects[obj.idx]
        return labels

    def add_frame_with_gt_seg(self, rgb: np.ndarray, depth: np.ndarray, segmentation: np.ndarray, fov: float, camera_ext: np.ndarray, gt_seg_entity_idx_to_info):
        self.new_objects.clear()
        self.curr_objects.clear()

        obs_seg_unique_ids = np.unique(segmentation).tolist()
        box_tags, box_names, boxes, masks = [], [], [], []
        for id in obs_seg_unique_ids:
            if id == -1:
                continue
            tag = gt_seg_entity_idx_to_info[id]["type"]
            name = gt_seg_entity_idx_to_info[id]["name"]
            if tag == "structure":
                continue
            box_tags.append(f"{tag}_{id}")
            box_names.append(f"{name}")
            mask = segmentation == id
            boxes.append([mask.any(axis=0).argmax(), mask.any(axis=1).argmax(), mask.any(axis=0).shape[0] - mask.any(axis=0)[::-1].argmax(), mask.any(axis=1).shape[0] - mask.any(axis=1)[::-1].argmax()])
            masks.append(mask)
        labels = -np.ones_like(depth, dtype=np.int32) # -1 - background -100 - Unknown

        if len(boxes) > 0:

            appearances = [ObjectBuilder._crop_image(rgb, box, mask) for box, mask in zip(boxes, masks)]

            start = time.time()
            areas = []
            cur_objects: list[Object] = []
            for name, tag, mask, appearance in zip(box_names, box_tags, masks, appearances):
                mask = mask & (depth < self.conf.depth_bound)
                if mask.sum() / mask.size < self.conf.detection_min_pixel_ratio:
                    areas.append(0)
                    cur_objects.append(None)
                    continue
                label = np.where(mask, 0, -100)
                points, colors, _ = VolumeGridBuilder._img_to_pcd(rgb, depth, label, fov, camera_ext)
                obj = Object(0, points, colors, appearance, tag, None, self.conf, self.num_frames, name=name)
                if obj.denoise(self.conf.denoise_param) < 5:
                    areas.append(0)
                    cur_objects.append(None)
                    continue

                areas.append(mask.sum())
                min_bound, max_bound = obj.get_bound()
                self.logger.debug(f"Detect object: {obj}")
                cur_objects.append(obj)

            # Step 2: Merge current objects with existing objects
            for cur_obj, appearance, mask in zip(cur_objects, appearances, masks):
                if cur_obj is None:
                    continue
                
                objects = [obj for obj in self.objects.values()]

                if len(objects) == 0:
                    self._new_object(cur_obj)
                    self.curr_objects.append(cur_obj.idx)
                    labels[mask] = cur_obj.idx
                    continue

                merge_obj_idx = None
                for obj in objects:
                    if obj.tag == cur_obj.tag:
                        merge_obj_idx = obj.idx
                        break

                if merge_obj_idx is not None:
                    self.logger.debug(f"Merge object: {cur_obj} into {objects[merge_obj_idx]}")

                    objects[merge_obj_idx].add_frame(rgb, depth, fov, camera_ext, mask, cur_obj.image_ft, self.num_frames, appearance)
                    objects[merge_obj_idx].denoise(self.conf.denoise_param)
                    labels[mask] = objects[merge_obj_idx].idx
                    self.curr_objects.append(objects[merge_obj_idx].idx)
                    del cur_obj
                else:
                    self._new_object(cur_obj)
                    self.curr_objects.append(cur_obj.idx)
                    labels[mask] = cur_obj.idx

            self.logger.debug(f"Object process time: {start}, {time.time()}")

        self.num_frames += 1
        return labels

    def add_frame_for_cur_objects(self, rgb: np.ndarray, depth: np.ndarray, fov: float, camera_ext: np.ndarray):
        self.new_objects.clear()
        self.curr_objects.clear()

        labels = -np.ones_like(depth, dtype=np.int32) # -1 - background -100 - Unknown
        # Step 1: Detect objects
        start = time.time()
        tags = self.ram.predict(rgb)
        self.logger.debug(f"RAM time: {start}, {time.time()}")
        tags = [tag for tag in tags if not self._ban_tag(tag)]
        tags.extend(["building", "person"])
        tags = list(set(tags))
        if self.debug:
            with open(os.path.join(self.output_path, "debug", f"{self.num_frames:06d}_tag.txt"), "w") as f:
                f.write(str(tags))
            boxes, box_tags, annotate = self.dino.predict(rgb, tags, annotate=True)
            Image.fromarray(annotate).save(os.path.join(self.output_path, "debug", f"{self.num_frames:06d}_dino.png"))
            if boxes.shape[0] > 0:
                masks, annotate = self.sam.predict(rgb, boxes, annotate=True)
                Image.fromarray(annotate.transpose(1, 0, 2, 3).reshape(rgb.shape[0], -1, 3)).save(os.path.join(self.output_path, "debug", f"{self.num_frames:06d}_sam.png"))
        else:
            start = time.time()
            boxes, box_tags = self.dino.predict(rgb, tags)
            self.logger.debug(f"DINO time: {start}, {time.time()}")
            if boxes.shape[0] > 0:
                start = time.time()
                masks = self.sam.predict(rgb, boxes)
                self.logger.debug(f"SAM time: {start}, {time.time()}")
        
        cur_objects: list[Object] = []
        if boxes.shape[0] > 0:
            # ensure masks are not overlapped
            # acc_mask = np.zeros_like(masks[0])
            # for i in range(boxes.shape[0]):
            #     masks[i] &= (depth < self.conf.depth_bound)
            #     masks[i] &= ~acc_mask
            #     acc_mask |= masks[i]
            
            appearances = [ObjectBuilder._crop_image(rgb, box, mask) for box, mask in zip(boxes, masks)]
            start = time.time()
            image_fts = self.clip.predict_image(appearances)
            self.logger.debug(f"CLIP time: {start}, {time.time()}")

            start = time.time()
            areas = []
            for box, tag, mask, image_ft, appearance in zip(boxes, box_tags, masks, image_fts, appearances):
                mask = mask & (depth < self.conf.depth_bound)
                if self._ban_tag(tag) or mask.sum() / mask.size < self.conf.detection_min_pixel_ratio:
                    areas.append(0)
                    cur_objects.append(None)
                    continue
                label = np.where(mask, 0, -100)
                points, colors, _ = VolumeGridBuilder._img_to_pcd(rgb, depth, label, fov, camera_ext)
                # TODO: denoise points
                min_bound, max_bound = points.min(axis=0), points.max(axis=0)
                if self.debug: self.logger.info(f"Detect object: {tag}, {min_bound}, {max_bound}")
                areas.append(mask.sum())
                cur_objects.append(Object(0, points, colors, appearance, tag, image_ft, self.conf, self.num_frames))
            
            for cur_obj, appearance, mask in zip(cur_objects, appearances, masks):
                if cur_obj is None:
                    continue

                # distinguish building(since it's large) and other objects
                if cur_obj.tag in BUILDING_TAGS:
                    objects = [obj for obj in self.objects.values() if obj.tag in BUILDING_TAGS]
                else:
                    objects = [obj for obj in self.objects.values() if not obj.tag in BUILDING_TAGS]

                if len(objects) == 0:
                    self._new_object(cur_obj)
                    self.curr_objects.append(cur_obj.idx)
                    labels[mask] = cur_obj.idx
                    continue
                
                sim = self._compute_sim(cur_obj, objects)
                merge_obj_idx = sim.argmax()

                if sim[merge_obj_idx] > self.conf.threshold:
                    self.logger.debug(f"Merge object: {cur_obj} into {objects[merge_obj_idx]}")

                    objects[merge_obj_idx].add_frame(rgb, depth, fov, camera_ext, mask, cur_obj.image_ft, self.num_frames, appearance)
                    objects[merge_obj_idx].denoise(self.conf.denoise_param)
                    labels[mask] = objects[merge_obj_idx].idx
                    self.curr_objects.append(objects[merge_obj_idx].idx)
                    del cur_obj
                else:
                    self._new_object(cur_obj)
                    self.curr_objects.append(cur_obj.idx)
                    labels[mask] = cur_obj.idx

        # print("object builder objects:")
        # for obj in self.objects.values():
        #     print(obj.tag)
        return labels, [obj for obj in cur_objects if obj is not None]
    
    def _compute_sim(self, cur_obj: Object, others: list[Object]):
        visual_sim = [np.dot(obj.image_ft, cur_obj.image_ft) for obj in others]
        spatial_sim = [cur_obj.get_overlap(obj) for obj in others] # is this single-direction ratio correct?

        self.logger.debug(f"Visual Similarity: {visual_sim}\nSpatial Similarity: {spatial_sim}")

        if cur_obj.tag in DYNAMIC_OBJECTS:
            # 0.88 as the solely visual similarity threshold -> 0.7 as final threshold
            sim = np.array(visual_sim) * 0.8 + np.array(spatial_sim) * 0.2
            for i, obj in enumerate(others):
                # should not merge dynamic objects detected in the same frame
                if cur_obj.tag != obj.tag or cur_obj.last_detected_frame == obj.last_detected_frame:
                    sim[i] = 0
        else:
            sim = np.array(visual_sim) * 0.5 + np.array(spatial_sim) * 0.5
            if sim.max() < self.conf.threshold:
                for i, obj in enumerate(others):
                    if visual_sim[i] > 0.88 and cur_obj.tag == obj.tag and spatial_sim[i] > 0.01:
                        sim[i] = 1.0
                        break
        self.logger.debug(f"Adjusted Similarity for {'dynamic' if cur_obj.tag in DYNAMIC_OBJECTS else 'static'} object of tag {cur_obj.tag}: {sim}")
        
        return sim
    
    def _new_object(self, obj: Object):
        self.logger.debug(f"New object detected: {obj}")
        obj.idx = max(self.objects.keys(), default=-1) + 1
        self.objects[obj.idx] = obj
        self.new_objects.append(obj.idx)
    
    def get_new_objects(self):
        new_objects = []
        for idx in self.new_objects:
            if idx in self.objects:
                new_objects.append(self.objects[idx])
        return new_objects
    
    def get_curr_objects(self):
        curr_objects = []
        for idx in self.curr_objects:
            if idx in self.objects:
                curr_objects.append(self.objects[idx])
        return curr_objects
    
    def save(self, path: str):
        for obj in self.objects.values():
            obj.save(os.path.join(self.output_path, f"obj_{obj.idx:06d}"))
        atomic_save(path, pickle.dumps(self.objects))
    
    def load(self, path: str):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.objects = pickle.load(f)
        # import pdb; pdb.set_trace()
        for obj in self.objects.values():
            self.num_frames = max(self.num_frames, obj.created_frame)
            obj.load(os.path.join(os.path.dirname(path), f"obj_{obj.idx:06d}"))
    
    def visualize(self):
        self.logger.info(f"Number of objects: {len(self.objects)}")
        for obj in self.objects.values():
            obj.visualize()

if __name__ == "__main__":
    logger = logging.Logger("ObjectBuilder")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    ob = ObjectBuilder(ObjectBuilderConfig(debug=True, logger=logger))
    ob.load("output/object/objects.pkl")
    # import pdb; pdb.set_trace()
    print([obj.idx for obj in ob.objects.values()])
    print(ob._compute_sim(ob.objects[1], [ob.objects[3]]))

