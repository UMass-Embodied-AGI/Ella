import os
import time
import numpy as np
from PIL import Image

import genesis as gs
from genesis.humanoid.humanoid_controller import HumanoidController
from .builder import Builder, BuilderConfig

FOV = 120

def main():
    ########################## init ##########################
    gs.init(seed=0, precision='32', logging_level='info', backend=gs.cpu)

    ########################## create a scene ##########################
    # p_trans = np.array([-130, -92.5, 4])
    # p_trans = np.array([-115.0, 0.0, 3.5])
    p_trans = np.array([0.0, 0.0, -1.0])

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            res=(960, 960),
            camera_pos=np.array([15.0, 3.0, 3.0]) + p_trans,
            camera_lookat=np.array([-15.0, -3.0, 0.0]) + p_trans,
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame = False,
            lights = [{'type': 'directional',
            'dir' : (0,0,-1),
            'color' : (1.0,1.0,1.0),
            'intensity' : 10.0},]
        ),
        sim_options=gs.options.SimOptions(
            gravity=(0, 0, 0),
        ),
        renderer=gs.renderers.Rasterizer(
        ),
    )

    ########################## entities ##########################
    mat_avatar = gs.materials.Avatar()
    mat_rigid = gs.materials.Rigid()

    controller1 = scene.add_humanoid(
        motion_data_path='motions/motion.pkl',
        skin_options={
            'glb_path': 'avatars/models/mixamo_Brian_merge.glb',
            'euler': (180, 0, 90),
            'pos': (0, 0, -1.0)
        },
        ego_view_options={
            "res": (256, 256),
            "fov": FOV,
            "GUI": True,
            "far": 1500,
        },
        frame_ratio=0.5,
    )

    # controller2 = scene.add_humanoid(
    #     motion_data_path='motions/motion.pkl',
    #     skin_options={
    #         'glb_path': 'avatars/models/mixamo_Sophie_merge.glb',
    #         'euler': (90, 45, -45),
    #         'pos': (0, 0, -0.8)
    #     },
    #     ego_view_options={
    #         "res": (256, 256),
    #         "fov": FOV,
    #         "GUI": True,
    #         "far": 1500,
    #     },
    # )

    ########################## city ##########################
    
    mesh = "/home/wangzeyuan/work/group/Ella/Ella/Genesis-dev/genesis/assets/scene/flat_CMU_mini_0704_highres.glb"
    mat_rigid = gs.materials.Rigid()    # glb
    gs.logger.info(f"Adding building {os.path.basename(mesh)}.")
    scene.add_entity(
            material=mat_rigid,
            morph=gs.morphs.Mesh(
                file=mesh,
				# euler=(90.0, 0, 0),
				fixed=True
            ),
            surface=gs.surfaces.Rough()
        )

    ########################## build ##########################
    
    scene.build()
    scene.reset()

    # controller1.turn_left(180)
    # while not controller1.spare():
    #     scene.step()
    # controller2.reset(np.array([6.0, -15.0, 0.0]) + p_trans)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    builder = Builder(BuilderConfig(debug=True))
    # builder.volume_grid_builder.load("output/point_cloud.pkl")
    controller1.reset(p_trans, np.diag([-1.0, -1.0, 1.0]))
    # controller2.reset(p_trans + np.array([2.0, 0, 0]))
    turn_around(scene, controller1, builder, output_dir)

def turn_around(scene, controller1: HumanoidController, builder: Builder, output_dir):
    controller1.turn_left(180)
    cnt = 0
    render_time, update_time = 0, 0
    while not controller1.spare():
        start_time = time.time()
        scene.step()

        # cur_trans = controller1.get_global_xy()
        # height = builder.volume_grid_builder.get_height(*cur_trans, 0.2)
        # if height is not None:
        #     controller1.set_global_height(height)

        if cnt % 3 == 0:
            rotation_offset=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        elif cnt % 3 == 1:
            rotation_offset=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) # left
        else:
            rotation_offset = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) # right

        rgb, dep, _ = controller1.render_ego_view(depth=True, rotation_offset=rotation_offset)
        render_time += time.time() - start_time
        Image.fromarray(rgb).save(os.path.join(output_dir, f"{cnt}.png"))

        start_time = time.time()
        builder.add_frame(rgb, dep, FOV, controller1.ego_view.extrinsics)
        update_time += time.time() - start_time
        cnt += 1
    print("Render time:", render_time / cnt, ", Update time:", update_time / cnt)
    print("Size:", builder.volume_grid_builder.get_size())
    print("Memory Size:", builder.volume_grid_builder.get_memory_size())
    builder.volume_grid_builder.save(os.path.join(output_dir, "point_cloud.pkl"))
    builder.volume_grid_builder.visualize()
    builder.object_builder.visualize()

def straight_walk(scene, controller1: HumanoidController, builder: Builder, output_dir):
    controller1.walk(10)
    cnt = 0
    render_time, update_time = 0, 0
    while not controller1.spare():
        start_time = time.time()
        scene.step()

        # cur_trans = controller1.get_global_xy()
        # height = builder.volume_grid_builder.get_height(*cur_trans, 0.2)
        # if height is not None:
        #     controller1.set_global_height(height)

        if cnt % 3 == 0:
            rotation_offset=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        elif cnt % 3 == 1:
            rotation_offset=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) # left
        else:
            rotation_offset = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) # right

        rgb, dep, _ = controller1.render_ego_view(depth=True, rotation_offset=rotation_offset)
        render_time += time.time() - start_time
        Image.fromarray(rgb).save(os.path.join(output_dir, f"{cnt}.png"))

        start_time = time.time()
        builder.add_frame(rgb, dep, FOV, controller1.ego_view.extrinsics)
        update_time += time.time() - start_time
        cnt += 1
    print("Render time:", render_time / cnt, ", Update time:", update_time / cnt)
    print("Size:", builder.volume_grid_builder.get_size())
    print("Memory Size:", builder.volume_grid_builder.get_memory_size())
    builder.volume_grid_builder.save(os.path.join(output_dir, "point_cloud.pkl"))
    builder.volume_grid_builder.visualize()
    builder.object_builder.visualize()

def goto(scene, controller1: HumanoidController, builder: Builder, output_dir, target: np.ndarray):
    cur_trans = np.array(controller1.get_global_xy())
    R = controller1.robot.global_rot
    cur_rad = np.arctan2(R[1, 0], R[0, 0])
    target_rad = np.arctan2(target[1] - cur_trans[1], target[0] - cur_trans[0])
    delta_rad = target_rad - cur_rad
    if delta_rad > np.pi:
        delta_rad -= 2 * np.pi
    elif delta_rad < -np.pi:
        delta_rad += 2 * np.pi
    
    if delta_rad > 0:
        controller1.turn_left(np.rad2deg(delta_rad))
    else:
        controller1.turn_right(np.rad2deg(-delta_rad))
    while not controller1.spare():
        scene.step()
        rgb, dep, _ = controller1.render_ego_view(depth=True)
        if builder is not None:
            builder.add_frame(rgb, dep, FOV, controller1.ego_view.extrinsics)
    
    controller1.walk(np.linalg.norm(target - cur_trans))
    cnt = 0
    while not controller1.spare():
        if builder is not None:
            height = builder.volume_grid_builder.get_height(*controller1.get_global_xy(), 0.2)
            if height is not None:
                controller1.set_global_height(height)
        scene.step()
        if cnt % 3 == 0:
            rotation_offset=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        elif cnt % 3 == 1:
            rotation_offset=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) # left
        else:
            rotation_offset = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) # right
        cnt += 1
        rgb, dep, _ = controller1.render_ego_view(rotation_offset=rotation_offset, depth=True)
        if builder is not None:
            builder.add_frame(rgb, dep, FOV, controller1.ego_view.extrinsics)

def navigate_to(scene, controller1: HumanoidController, builder: Builder, output_dir, goal: np.ndarray):
    last_trans = np.array(controller1.get_global_xy())
    while True:
        cur_trans = np.array(controller1.get_global_xy())
        if np.linalg.norm(goal - cur_trans) < 5:
            break
        start_time = time.time()
        path = builder.volume_grid_builder.navigate(cur_trans, goal)
        print("Navigate time:", time.time() - start_time)
        if path is None:
            cur_goal = last_trans # go back
        else:
            cur_goal = path[min(10, len(path) - 1)]
        print("Current goal:", cur_goal, ", Final goal:", goal)
        goto(scene, controller1, builder, output_dir, cur_goal)
        last_trans = cur_trans
        
        print("Size:", builder.volume_grid_builder.get_size())
        print("Memory Size:", builder.volume_grid_builder.get_memory_size())
        builder.volume_grid_builder.save(os.path.join(output_dir, "point_cloud.pkl"))
    
    builder.volume_grid_builder.visualize()

def random_walk(scene, controller1: HumanoidController, builder: Builder, output_dir):
    last_trans = np.array(controller1.get_global_xy())
    goal = np.array([-60.0, 0.0])
    
    while True:
        cur_trans = np.array(controller1.get_global_xy())
        if np.linalg.norm(goal - cur_trans) < 5:
            break
        start_time = time.time()
        path = builder.volume_grid_builder.navigate(cur_trans, goal)
        print("Navigate time:", time.time() - start_time)
        if path is None:
            cur_goal = last_trans # go back
        else:
            cur_goal = path[min(10, len(path) - 1)]
        print("Current goal:", cur_goal, ", Final goal:", goal)
        goto(scene, controller1, builder, output_dir, cur_goal)
        last_trans = cur_trans
        
        print("Size:", builder.volume_grid_builder.get_size())
        print("Memory Size:", builder.volume_grid_builder.get_memory_size())
        builder.volume_grid_builder.save(os.path.join(output_dir, "point_cloud.pkl"))
    
    builder.volume_grid_builder.visualize()

def infinity_walk(scene, controller1: HumanoidController, builder: Builder, output_dir):
    # time.sleep(10000)
    pa, pb = np.array([-120.0, 0.0]), np.array([120.0, 0.0])
    goto(scene, controller1, builder, output_dir, pb)
    # goto(scene, controller1, builder, output_dir, pb)
    # builder.object_builder.visualize()

if __name__ == '__main__':
    main()