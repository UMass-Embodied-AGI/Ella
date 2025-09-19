import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import genesis as gs
from genesis.utils.mesh import *
import pygltflib
import trimesh
from tqdm import tqdm
import ctypes
from sklearn.cluster import SpectralClustering
from PIL import Image, ImageDraw
from .sg.builder.volume_grid import VolumeGridBuilder
import json

lib_utils = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'utils.so'))
lib_utils.smooth.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8)]
lib_utils.smooth.restype = None

lib_utils.bfs.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
lib_utils.bfs.restype = None

lib_utils.adj_matrix.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
lib_utils.adj_matrix.restype = None

lib_utils.flood_fill.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
lib_utils.flood_fill.restype = None

@dataclass
class Mesh2:
    verts: np.ndarray
    faces: np.ndarray

def parse_mesh_glb2(path, group_by_material, scale, surface):
    glb = pygltflib.GLTF2().load(path)
    assert glb is not None

    def parse_tree(node_index):
        node = glb.nodes[node_index]
        if node.matrix is not None:
            matrix = np.array(node.matrix, dtype=float).reshape((4, 4))
        else:
            matrix = np.identity(4, dtype=float)
            if node.translation is not None:
                translation = np.array(node.translation, dtype=float)
                translation_matrix = np.identity(4, dtype=float)
                translation_matrix[3, :3] = translation
                matrix = translation_matrix @ matrix
            if node.rotation is not None:
                rotation = np.array(node.rotation, dtype=float)     # xyzw
                rotation_matrix = np.identity(4, dtype=float)
                rotation = [rotation[3], rotation[0], rotation[1], rotation[2]]
                rotation_matrix[:3, :3] = trimesh.transformations.quaternion_matrix(rotation)[:3, :3].T
                matrix = rotation_matrix @ matrix
            if node.scale is not None:
                scale = np.array(node.scale, dtype=float)
                scale_matrix = np.diag(np.append(scale, 1))
                matrix = scale_matrix @ matrix
        mesh_list = list()
        if node.mesh is not None:
            mesh_list.append([node.mesh, np.identity(4, dtype=float)])
        for sub_node_index in node.children:
            sub_mesh_list = parse_tree(sub_node_index)
            mesh_list.extend(sub_mesh_list)
        for i in range(len(mesh_list)):
            mesh_list[i][1] = mesh_list[i][1] @ matrix
        return mesh_list

    def get_bufferview_data(buffer_view):
        buffer = glb.buffers[buffer_view.buffer]
        return glb.get_data_from_buffer_uri(buffer.uri)

    def get_data_from_accessor(accessor_index):
        accessor = glb.accessors[accessor_index]
        buffer_view = glb.bufferViews[accessor.bufferView]
        buffer_data = get_bufferview_data(buffer_view)

        data_type, data_ctype, count = accessor.type, accessor.componentType, accessor.count
        dtype = ctype_to_numpy[data_ctype][1]
        itemsize = np.dtype(dtype).itemsize
        buffer_byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
        num_components = type_to_count[data_type][0]

        byte_stride = buffer_view.byteStride if buffer_view.byteStride else num_components * itemsize
        # Extract data considering byteStride
        if byte_stride == num_components * itemsize:
            # Data is tightly packed
            byte_length = count * num_components * itemsize
            data = buffer_data[buffer_byte_offset : buffer_byte_offset + byte_length]
            array = np.frombuffer(data, dtype=dtype)
            if num_components > 1:
                array = array.reshape((count, num_components))
        else:
            # Data is interleaved
            array = np.zeros((count, num_components), dtype=dtype)
            for i in range(count):
                start = buffer_byte_offset + i * byte_stride
                end = start + num_components * itemsize
                data_slice = buffer_data[start:end]
                array[i] = np.frombuffer(data_slice, dtype=dtype, count=num_components)

        return array.reshape([count] + type_to_count[data_type][1])

    glb.convert_images(pygltflib.ImageFormat.DATAURI)

    scene = glb.scenes[glb.scene]
    mesh_list = list()
    for node_index in scene.nodes:
        root_mesh_list = parse_tree(node_index)
        mesh_list.extend(root_mesh_list)

    temp_infos = dict()
    for i in tqdm(range(len(mesh_list))):
        mesh = glb.meshes[mesh_list[i][0]]
        matrix = mesh_list[i][1]
        for primitive in mesh.primitives:
            if group_by_material:
                group_idx = primitive.material
            else:
                group_idx = i

            uvs0, uvs1 = None, None
            if 'KHR_draco_mesh_compression' in primitive.extensions:
                import DracoPy
                KHR_index = primitive.extensions['KHR_draco_mesh_compression']['bufferView']
                mesh_buffer_view = glb.bufferViews[KHR_index]
                mesh_data = get_bufferview_data(mesh_buffer_view)
                mesh = DracoPy.decode(mesh_data[
                                      mesh_buffer_view.byteOffset:
                                      mesh_buffer_view.byteOffset + mesh_buffer_view.byteLength
                                      ])
                points = mesh.points
                triangles = mesh.faces
                normals = mesh.normals if len(mesh.normals) > 0 else None
                uvs0 = mesh.tex_coord if len(mesh.tex_coord) > 0 else None

            else:
                # "primitive.attributes" records accessor indices in "glb.accessors", like:
                #      Attributes(POSITION=2, NORMAL=1, TANGENT=None, TEXCOORD_0=None, TEXCOORD_1=None,
                #                 COLOR_0=None, JOINTS_0=None, WEIGHTS_0=None)
                # parse vertices

                points = get_data_from_accessor(primitive.attributes.POSITION).astype(float)

                if primitive.indices is None:
                    indices = np.arange(points.shape[0], dtype=np.uint32)
                else:
                    indices = get_data_from_accessor(primitive.indices).astype(np.int32)

                mode = primitive.mode if primitive.mode is not None else 4

                if mode == 4:  # TRIANGLES
                    triangles = indices.reshape(-1, 3)
                elif mode == 5:  # TRIANGLE_STRIP
                    triangles = []
                    for i in range(len(indices) - 2):
                        if i % 2 == 0:
                            triangles.append([indices[i], indices[i + 1], indices[i + 2]])
                        else:
                            triangles.append([indices[i], indices[i + 2], indices[i + 1]])
                    triangles = np.array(triangles, dtype=np.uint32)
                elif mode == 6:  # TRIANGLE_FAN
                    triangles = []
                    for i in range(1, len(indices) - 1):
                        triangles.append([indices[0], indices[i], indices[i + 1]])
                    triangles = np.array(triangles, dtype=np.uint32)
                else:
                    gs.logger.warning(f"Primitive mode {mode} not supported.")
                    continue  # Skip unsupported modes


                # parse normals
                if primitive.attributes.NORMAL:
                    normals = get_data_from_accessor(primitive.attributes.NORMAL).astype(float)
                else:
                    normals = None

                # parse uvs
                if primitive.attributes.TEXCOORD_0:
                    uvs0 = get_data_from_accessor(primitive.attributes.TEXCOORD_0).astype(float)
                if primitive.attributes.TEXCOORD_1:
                    uvs1 = get_data_from_accessor(primitive.attributes.TEXCOORD_1).astype(float)

            if normals is None:
                normals = trimesh.Trimesh(points, triangles, process=False).vertex_normals
            points, normals = apply_transform(matrix, points, normals)

            if group_idx not in temp_infos.keys():
                temp_infos[group_idx] = {
                    'mat_index' : primitive.material,
                    'points'    : [points],
                    'triangles' : [triangles],
                    'normals'   : [normals],
                    'uvs0'      : [uvs0],
                    'uvs1'      : [uvs1],
                    'n_points'  : len(points),
                }

            else:
                triangles += temp_infos[group_idx]['n_points']
                temp_infos[group_idx]['points'].append(points)
                temp_infos[group_idx]['triangles'].append(triangles)
                temp_infos[group_idx]['normals'].append(normals)
                temp_infos[group_idx]['uvs0'].append(uvs0)
                temp_infos[group_idx]['uvs1'].append(uvs1)
                temp_infos[group_idx]['n_points'] += len(points)

    meshes = list[Mesh2]()
    for group_idx in tqdm(temp_infos.keys()):
        # build other group properties
        verts   = np.concatenate(temp_infos[group_idx]['points'])
        normals = np.concatenate(temp_infos[group_idx]['normals'])
        faces   = np.concatenate(temp_infos[group_idx]['triangles'])

        meshes.append(
            Mesh2(verts, faces)
        )

    return meshes

N = 450

def draw_line(occ_map: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    p1 = p1 + N
    p2 = p2 + N
    l = np.linalg.norm(p2 - p1)
    for i in range(int(l * 2)):
        p = p1 + (p2 - p1) * i / l / 2
        if 0 <= p[0] < occ_map.shape[1] and 0 <= p[1] < occ_map.shape[0]:
            occ_map[int(p[1]), int(p[0])] = 1

def proj_xy(p: np.ndarray):
    return np.array([p[0], p[2]])

def region_cluster(occ_map: np.ndarray):
    dist = np.zeros(occ_map.shape, dtype=np.int32)
    id = np.zeros(occ_map.shape, dtype=np.int32)
    lib_utils.bfs(*occ_map.shape,
                  occ_map.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                  id.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                  dist.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    max_id = np.max(id)
    mat = np.zeros((max_id, max_id), dtype=np.int32)
    lib_utils.adj_matrix(*occ_map.shape,
                         id.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                         dist.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                         mat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    Image.fromarray(((dist >= 0) * 255).astype(np.uint8)).save("dist.png")
    # mat = mat.clip(1)
    # plt.imshow(dist == 0, cmap='gray')
    mat = mat.astype(np.float64)
    n_clusters = int(np.sqrt(max_id))
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed_nearest_neighbors', n_neighbors=max(2, max_id // (n_clusters * 2)))
    label = clustering.fit_predict(mat ** 2)
    # G = nx.Graph(mat)
    # partition = community.best_partition(G)
    # label = np.array([partition[i] for i in range(max_id)])
    colors = np.random.rand(label.max() + 1, 3)
    img = colors[label[id - 1]]
    img[occ_map.astype(bool)] = 0
    Image.fromarray((img * 255).astype(np.uint8)).save("output.png")
    # img = colors[id - 1]
    # img[dist == 0] = 0
    # plt.imshow(img)
    # plt.savefig("output.png")

if __name__ == "__main__":
    gs.init()
    occ_map = np.zeros((N * 2, N * 2), dtype=np.uint8)
    glb_path = "Genesis/genesis/assets/ViCo/scene/final/DETROIT_ok/buildings_basic.glb"
    meshes: list[Mesh2] = parse_mesh_glb2(glb_path, False, 1, None)
    for mesh in tqdm(meshes):
        for f in mesh.faces:
            draw_line(occ_map, proj_xy(mesh.verts[f[0]]), proj_xy(mesh.verts[f[1]]))
    # vg = VolumeGridBuilder()
    # vg.load("output/volume_grid_4700.pkl")
    # occ_map: np.ndarray = (vg.get_occ_map() == 2).astype(np.uint8)
    # lib_utils.smooth(*occ_map.shape, 1, occ_map.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))

    with open("sg/test/old_assets/scratch.json", "r") as f:
        schedule = json.load(f)
    with open("sg/test/old_assets/DETROIT/building_metadata.json", "r") as f:
        building_metadata = json.load(f)
    poses = [np.mean(building_metadata[s["building"]]["bounding_box"], axis=0) for s in schedule if s["building"] is not None]

    vg = VolumeGridBuilder()
    vg.load("output/volume_grid.pkl")
    agent_occ_map, x_min, y_min, x_max, y_max = vg.get_occ_map()
    
    Image.fromarray((occ_map * 255).astype(np.uint8)).save("occ_map.png")

    filled_map = np.zeros(occ_map.shape, dtype=np.int32)
    lib_utils.flood_fill(*occ_map.shape, occ_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), filled_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    print(filled_map.shape, filled_map.sum())
    image = np.zeros((N * 2, N * 2, 3), dtype=np.uint8)
    for y in range(agent_occ_map.shape[0]):
        for x in range(agent_occ_map.shape[1]):
            if agent_occ_map[y, x] == 1:
                continue
            image[N - (y + y_min) // 2, N + (x + x_min) // 2] = [255, 0, 0]
    image[filled_map == 0] = [255, 255, 255]
    image = Image.fromarray(image)
    # Draw green circles for each pose

    draw = ImageDraw.Draw(image)
    for pose in poses:
        # Convert pose coordinates to image coordinates
        x = int(N + pose[0])
        y = int(N - pose[1])
        
        # Draw green circle with radius 5
        draw.ellipse((x-5, y-5, x+5, y+5), fill=(0, 255, 0), outline=(0, 255, 0))
    image.save("filled_map.png")
