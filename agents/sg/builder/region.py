from logging import Logger
import os
import numpy as np
import ctypes
from sklearn.cluster import SpectralClustering
from PIL import Image
import json

from .builtin import lib_region
from .volume_grid import VolumeGridBuilder
from .object import ObjectBuilder
from vico.tools.utils import atomic_save

class RegionBuilder:
    def __init__(self, vg_builder: VolumeGridBuilder, obj_builder: ObjectBuilder, logger: Logger = None, debug = False, output_dir = None):
        self.vg_builder = vg_builder
        self.obj_builder = obj_builder
        self.logger = logger
        self.debug = debug
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.region = {}
    
    def add_frame(self):
        occ_map, x_min, y_min, x_max, y_max = self.vg_builder.get_occ_map()
        occ_map: np.ndarray = (occ_map == 2).astype(np.uint8)
        lib_region.smooth(*occ_map.shape, 1, occ_map.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        # Image.fromarray((occ_map * 255).astype(np.uint8)).save("occ_map.png")

        dist = np.zeros(occ_map.shape, dtype=np.int32)
        id = np.zeros(occ_map.shape, dtype=np.int32)
        lib_region.bfs(*occ_map.shape,
                    occ_map.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    id.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    dist.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        max_id = np.max(id)
        n_clusters = int(np.sqrt(max_id))
        if n_clusters == 0:
            return
        
        mat = np.zeros((max_id, max_id), dtype=np.int32)
        lib_region.adj_matrix(*occ_map.shape,
                            id.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                            dist.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                            mat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        # Image.fromarray(((dist >= 0) * 255).astype(np.uint8)).save("dist.png")
        # mat = mat.clip(1)
        # plt.imshow(dist == 0, cmap='gray')
        mat = mat.astype(np.float64)
        if n_clusters > 1:
            clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed_nearest_neighbors', n_neighbors=max(2, max_id // (n_clusters * 2)))
            label = clustering.fit_predict(mat ** 2)
        else:
            label = np.zeros(max_id, dtype=np.int32)

        colors = np.random.rand(label.max() + 1, 3)
        img = colors[label[id - 1]]
        img[occ_map.astype(bool)] = 0
        if self.debug and self.output_dir:
            Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(self.output_dir, f"region_{self.obj_builder.num_frames}.png"))

        map_label = label[id - 1]
        for obj in self.obj_builder.objects.values():
            if obj.tag == "building":
                points = obj.volume_grid_builder.get_points()[0]
                points = obj.volume_grid_builder.align_nav(points).astype(np.int32)[:, :2]
                points[:, 0] -= x_min
                points[:, 1] -= y_min
                valid_mask = (points[:, 0] >= 0) & (points[:, 0] < occ_map.shape[1]) & (points[:, 1] >= 0) & (points[:, 1] < occ_map.shape[0])
                points = points[valid_mask]
                point_labels = map_label[points[:, 1], points[:, 0]]
                values, counts = np.unique(point_labels, return_counts=True)
                self.region[obj.idx] = int(values[np.argmax(counts)])
                if self.logger:
                    self.logger.critical(f"Put {obj.idx} into region {self.region[obj.idx]}")
    
    def save(self, path):
        atomic_save(path, json.dumps(self.region))
    
    def load(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.region = json.load(f)
