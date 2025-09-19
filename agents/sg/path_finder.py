from .sg.builder.builder import Builder
# from persona.cognitive_modules.converse2 import select_personas_to_converse
import genesis.utils.geom as geom_utils
import numpy as np
import os
import time

FOV = 120

def adjust_height(controller, builder: Builder, last_height: float):
    height = builder.volume_grid_builder.get_height(*controller.get_global_xy(), 0.2)
    if last_height is not None and (height is None or abs(height - last_height) > 0.5):
        # too large difference, ignore
        height = last_height
    controller.set_global_height(height)
    return height

def transform_q(base_trans, base_rot):
    '''Map normal 7+4X pose array to 7+7X floating joint version'''
    q = np.zeros(1 * 7) # n_link * (n_T + n_R)
    # for i in range(1):
    #     q[7+i*7+3:7+i*7+7] = [1,0,0,0]
    q[:3] = base_trans
    q[3:7] = geom_utils.R_to_quat(base_rot)
    return q

FOV = 120

def goto_all(scene, controllers, builder: Builder, output_dir, targets, tracing_cams):

    new_third_person_frames = [[] for _ in range(len(controllers))]

    cur_trans_list = []
    for i, controller1 in enumerate(controllers):
        cur_trans = np.array(controller1.get_global_xy())
        cur_trans_list.append(cur_trans)
        R = controller1.robot.global_rot
        cur_rad = np.arctan2(R[1, 0], R[0, 0])
        target_rad = np.arctan2(targets[i][1] - cur_trans[1], targets[i][0] - cur_trans[0])
        delta_rad = target_rad - cur_rad
        if delta_rad > np.pi:
            delta_rad -= 2 * np.pi
        elif delta_rad < -np.pi:
            delta_rad += 2 * np.pi
        if delta_rad > 0:
            controller1.turn_left(np.rad2deg(delta_rad))
        else:
            controller1.turn_right(np.rad2deg(-delta_rad))
    while not all([controller1.spare() for controller1 in controllers]):
        scene.step()
        for i, controller1 in enumerate(controllers):
            rgb, dep, _ = controller1.render_ego_view(depth=True)
            builder.add_frame(rgb, dep, FOV, controller1.ego_view.extrinsics)
            tracing_cams[i].set_pose(pos=np.array([cur_trans_list[i][0]-controller1.robot.global_rot[0, 0]*3.0,
                                            cur_trans_list[i][1]-controller1.robot.global_rot[1, 0]*3.0,
                                            controller1.get_global_height()+3.0]),
                                lookat=np.array([cur_trans_list[i][0],
                                                cur_trans_list[i][1],
                                                controller1.get_global_height()+3.0]))
            rgb_arr, _, _ = tracing_cams[i].render()
            new_third_person_frames[i].append(rgb_arr)
    for i, controller1 in enumerate(controllers):
        controller1.walk(np.linalg.norm(targets[i] - cur_trans_list[i]))
    cnt = 0
    last_height = [None for i in range(len(controllers))]
    while not all([controller1.spare() for controller1 in controllers]):
        scene.step()
        for i, controller1 in enumerate(controllers):
            last_height[i] = adjust_height(controller1, builder, last_height[i])
            if cnt % 3 == 0:
                rotation_offset=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            elif cnt % 3 == 1:
                rotation_offset=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            else:
                rotation_offset = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            cnt += 1
            cur_trans_new = np.array(controller1.get_global_xy())
            global_rot = np.eye(3)
            locator_step_q = transform_q(np.array([cur_trans_new[0], cur_trans_new[1], 50.0]), global_rot)
            # locating_spheres[i].set_q(locator_step_q)
            rgb, dep, _ = controller1.render_ego_view(rotation_offset=rotation_offset, depth=True)
            builder.add_frame(rgb, dep, FOV, controller1.ego_view.extrinsics)
            """
            R_z(theta) = [[cos(theta), -sin(theta), 0],[sin(theta), cos(theta), 0],[0, 0, 1]]
            """
            tracing_cams[i].set_pose(pos=np.array([cur_trans_new[0]-controller1.robot.global_rot[0, 0]*3.0,
                                            cur_trans_new[1]-controller1.robot.global_rot[1, 0]*3.0,
                                            controller1.get_global_height()+3.0]), 
                                lookat=np.array([cur_trans_new[0],
                                                cur_trans_new[1],
                                                controller1.get_global_height()+3.0]))
            rgb_arr, _, _ = tracing_cams[i].render()
            new_third_person_frames[i].append(rgb_arr)
    return new_third_person_frames

def random_walk_all2(scene, personas, controllers, builder: Builder, output_dir, goals, tracing_cams, only_nav=True):
    last_trans_list = []
    controllers_finishing_status = []
    for controller in controllers:
        last_trans_list.append(np.array(controller.get_global_xy()))
        controllers_finishing_status.append(False)
    
    third_person_frames = [[] for _ in range(len(controllers))]
    while True:
        print("controller finishing status:", controllers_finishing_status)
        cur_goals = []
        cur_trans_list = []
        for i, controller in enumerate(controllers):
            cur_trans = np.array(controller.get_global_xy())
            cur_trans_list.append(cur_trans)
            start_time = time.time()
            path = builder.volume_grid_builder.navigate(cur_trans, goals[i])
            # print("Navigate time:", time.time() - start_time)
            if path is None:
                cur_goal = last_trans_list[i] # go back
            else:
                cur_goal = path[min(10, len(path) - 1)]
            cur_goals.append(cur_goal)
            # print("Current goal:", cur_goal, ", Final goal:", goal)
            last_trans_list[i] = cur_trans
        for i, controller in enumerate(controllers):
            if np.linalg.norm(goals[i] - cur_trans_list[i]) < 15:
                controllers_finishing_status[i] = True
            else:
                print("unfinished controller delta dist:", np.linalg.norm(goals[i] - cur_trans_list[i]))
                # new_controllers.pop(i)
                # cur_goals.pop(i)
                # tracing_cams.pop(i)
                # locating_spheres.pop(i)
        new_third_person_frames = goto_all(scene, controllers, builder, output_dir, cur_goals, tracing_cams)
        if len(new_third_person_frames) == len(third_person_frames):
            for i in range(len(third_person_frames)):
                third_person_frames[i].extend(new_third_person_frames[i])
        # print("Size:", builder.volume_grid_builder.get_size())
        # print("Memory Size:", builder.volume_grid_builder.get_memory_size())
        # print("Average Render Time:", render_time / steps, ", Build Time:", build_time / steps)
        # builder.occ_map_builder.save(os.path.join(output_dir, "occ_map.pkl"))
        builder.volume_grid_builder.save(os.path.join(output_dir, "point_cloud.pkl"))
        # builder.volume_grid_builder.visualize()
        # builder.occ_map_builder.visualize()
        # builder.volume_grid_builder.visualize()
        for i, persona_name in enumerate(personas.keys()):
            cur_trans = np.array(controllers[i].get_global_xy())
            personas[persona_name].current_position = np.array([cur_trans[0], cur_trans[1], 0])

        if not only_nav:
            ### React to other people or events? ###
            for i, persona_name in enumerate(personas.keys()):
                have_conversation, new_third_person_frames = select_personas_to_converse(scene, persona_name, personas, controllers, tracing_cams)
                if have_conversation:
                    controllers_finishing_status[i] = True
                    if len(new_third_person_frames) == len(third_person_frames):
                        for i in range(len(third_person_frames)):
                            third_person_frames[i].extend(new_third_person_frames[i])
        if all(controllers_finishing_status):
            break
    return third_person_frames

def random_walk_single_agent(persona, controller, builder: Builder, output_dir, goal):
    cur_trans = np.array(controller.get_global_xy())
    start_time = time.time()
    path = builder.volume_grid_builder.navigate(cur_trans, goal)
    # print("Navigate time:", time.time() - start_time)
    if path is None:
        cur_goal = last_trans # go back
    else:
        cur_goal = path[min(10, len(path) - 1)]
    # print("Current goal:", cur_goal, ", Final goal:", goal)
    last_trans = cur_trans
    if np.linalg.norm(goal - cur_trans) < 15:
        persona.curr_nav_done = True
        return None
    else:
        print("unfinished controller delta dist:", np.linalg.norm(goal - cur_trans))
    # new_third_person_frames = goto_all(scene, controller, builder, output_dir, cur_goal, tracing_cam, locating_sphere)
    # print("Size:", builder.volume_grid_builder.get_size())
    # print("Memory Size:", builder.volume_grid_builder.get_memory_size())
    # print("Average Render Time:", render_time / steps, ", Build Time:", build_time / steps)
    # builder.occ_map_builder.save(os.path.join(output_dir, "occ_map.pkl"))
    builder.volume_grid_builder.save(os.path.join(output_dir, "point_cloud.pkl"))
    # builder.volume_grid_builder.visualize()
    # builder.occ_map_builder.visualize()
    # builder.volume_grid_builder.visualize()
    # persona.current_position = np.array([cur_trans[0], cur_trans[1], 0])
    return cur_goal

"""
Archiving old implementations
"""

# def goto(scene, controller1, builder: Builder, output_dir, target: np.ndarray, tracing_cam):
#     # print("enters goto")
#     cur_trans = np.array(controller1.get_global_xy())
#     R = controller1.robot.global_rot
#     cur_rad = np.arctan2(R[1, 0], R[0, 0])
#     target_rad = np.arctan2(target[1] - cur_trans[1], target[0] - cur_trans[0])
#     delta_rad = target_rad - cur_rad
#     if delta_rad > np.pi:
#         delta_rad -= 2 * np.pi
#     elif delta_rad < -np.pi:
#         delta_rad += 2 * np.pi

#     # height = builder.volume_grid_builder.get_height(*controller1.get_global_xy(), 0.2) # this is None
#     # print("here0")
#     if delta_rad > 0:
#         controller1.turn_left(np.rad2deg(delta_rad))
#     else:
#         controller1.turn_right(np.rad2deg(-delta_rad))
#     while not controller1.spare():
#         # print("here1")
#         scene.step()
#         rgb, dep, _ = controller1.render_ego_view(depth=True)
#         builder.add_frame(rgb, dep, FOV, controller1.ego_view.extrinsics)
#         tracing_cam.set_pose(pos=np.array([cur_trans[0]-controller1.robot.global_rot[0, 0]*3,
#                                            cur_trans[1]-controller1.robot.global_rot[1, 0]*3,
#                                            controller1.get_global_height()+3]),
#                             lookat=np.array([cur_trans[0],
#                                              cur_trans[1],
#                                              controller1.get_global_height()+3]))
#         tracing_cam.render()
    
#     controller1.walk(np.linalg.norm(target - cur_trans))
#     cnt = 0
#     while not controller1.spare():
#         # print("here2")
#         scene.step()
#         height = builder.volume_grid_builder.get_height(*controller1.get_global_xy(), 0.2)
#         if height is not None:
#             controller1.set_global_height(height)
#         if cnt % 3 == 0:
#             rotation_offset=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#         elif cnt % 3 == 1:
#             rotation_offset=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
#         else:
#             rotation_offset = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
#         cnt += 1
#         cur_trans_new = np.array(controller1.get_global_xy())
#         rgb, dep, _ = controller1.render_ego_view(rotation_offset=rotation_offset, depth=True)
#         builder.add_frame(rgb, dep, FOV, controller1.ego_view.extrinsics)
#         """
#         R_z(theta) = [[cos(theta), -sin(theta), 0],[sin(theta), cos(theta), 0],[0, 0, 1]]
#         """
#         # print("rdebug:", controller1.robot.global_rot[2][2], controller1.robot.global_rot[0][2])
#         tracing_cam.set_pose(pos=np.array([cur_trans_new[0]-controller1.robot.global_rot[0, 0]*3,
#                                            cur_trans_new[1]-controller1.robot.global_rot[1, 0]*3,
#                                            controller1.get_global_height()+3]), 
#                             lookat=np.array([cur_trans_new[0],
#                                              cur_trans_new[1],
#                                              controller1.get_global_height()+3]))
#         tracing_cam.render()


# def random_walk_all(scene, controllers, builder: Builder, output_dir, goals, tracing_cams):
#     last_trans_list = []
#     controllers_finishing_status = []
#     for controller in controllers:
#         last_trans_list.append(np.array(controller.get_global_xy()))
#         controllers_finishing_status.append(False)
    
#     while True:
#         for i, controller in enumerate(controllers):
#             cur_trans = np.array(controller.get_global_xy())
#             if np.linalg.norm(goals[i] - cur_trans) < 5:
#                 controllers_finishing_status[i] = True
#             start_time = time.time()
#             path = builder.volume_grid_builder.navigate(cur_trans, goals[i])
#             # print("Navigate time:", time.time() - start_time)
#             if path is None:
#                 cur_goal = last_trans_list[i] # go back
#             else:
#                 cur_goal = path[min(10, len(path) - 1)]
#             # print("Current goal:", cur_goal, ", Final goal:", goal)
#             goto(scene, controller, builder, output_dir, cur_goal, tracing_cams[i])
#             last_trans_list[i] = cur_trans
            
#             # print("Size:", builder.volume_grid_builder.get_size())
#             # print("Memory Size:", builder.volume_grid_builder.get_memory_size())
#             # print("Average Render Time:", render_time / steps, ", Build Time:", build_time / steps)
#             # builder.occ_map_builder.save(os.path.join(output_dir, "occ_map.pkl"))
#             builder.volume_grid_builder.save(os.path.join(output_dir, "point_cloud.pkl"))
#             # builder.volume_grid_builder.visualize()
#             # builder.occ_map_builder.visualize()
#             # builder.volume_grid_builder.visualize()
#         if all(controllers_finishing_status):
#             break

# def random_walk(scene, controller1, builder: Builder, output_dir, goal, tracing_cam):
#     last_trans = np.array(controller1.get_global_xy())
    
#     while True:
#         cur_trans = np.array(controller1.get_global_xy())
#         if np.linalg.norm(goal - cur_trans) < 5:
#             break
#         start_time = time.time()
#         path = builder.volume_grid_builder.navigate(cur_trans, goal)
#         # print("Navigate time:", time.time() - start_time)
#         if path is None:
#             cur_goal = last_trans # go back
#         else:
#             cur_goal = path[min(10, len(path) - 1)]
#         # print("Current goal:", cur_goal, ", Final goal:", goal)
#         goto(scene, controller1, builder, output_dir, cur_goal, tracing_cam)
#         last_trans = cur_trans
        
#         # print("Size:", builder.volume_grid_builder.get_size())
#         # print("Memory Size:", builder.volume_grid_builder.get_memory_size())
#         # print("Average Render Time:", render_time / steps, ", Build Time:", build_time / steps)
#         # builder.occ_map_builder.save(os.path.join(output_dir, "occ_map.pkl"))
#         builder.volume_grid_builder.save(os.path.join(output_dir, "point_cloud.pkl"))
#         # builder.volume_grid_builder.visualize()
#         # builder.occ_map_builder.visualize()
#         # builder.volume_grid_builder.visualize()