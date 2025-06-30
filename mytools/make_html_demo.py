import os
import re
from collections import defaultdict

import cv2
import sys
import json
import argparse
import subprocess
import numpy as np
from tqdm import tqdm

from PIL import Image, ImageDraw
import matplotlib.colors as mcolors

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
import sys
current_directory = os.getcwd()
sys.path.insert(0, current_directory)
from mytools.utils import merge_step_files

def map_lang_colors_to_rgb(lang_colors):
	rgb_list = []
	for lang_color in lang_colors:
		rgb_list.append(mcolors.to_rgb(lang_color))
	return rgb_list

# def render_topdown_locators(image, locator_positions, colors, circle_radii, camera_parameters):
#     f_x = camera_parameters["camera_res"][0] / (2.0 * np.tan(np.radians(camera_parameters["camera_fov"] / 2.0)))
#     f_y = camera_parameters["camera_res"][1] / (2.0 * np.tan(np.radians(camera_parameters["camera_fov"] / 2.0)))
#     intrinsic_K = np.array([[f_x, 0.0, camera_parameters["camera_res"][0]/2.0],
#                             [0.0, f_y, camera_parameters["camera_res"][1]/2.0],
#                             [0.0, 0.0, 1.0]])
#     extrinsic = np.array(camera_parameters["camera_extrinsics"])
#     extrinsic = extrinsic[:3, :4]
#     for pos, color, radius in zip(locator_positions, colors, circle_radii):
#         P_world = np.append(pos, 1.0)
#         P_camera = extrinsic @ P_world
#         P_image = intrinsic_K @ P_camera
#         pixel_x = int(P_image[0] / P_image[2])
#         pixel_y = int(P_image[1] / P_image[2])
#         cv2.circle(image, (pixel_x, pixel_y), radius, (color[2]*255, color[1]*255, color[0]*255), -1)
#     return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# def rgb_to_bgr255(color):
#     rgb = mcolors.to_rgb(color)
#     # gbr = (rgb[2]*255, rgb[1]*255, rgb[0]*255)
#     rgb = (rgb[0]*255, rgb[1]*255, rgb[2]*255)
#     return rgb

# def make_global_image(i, args, names_order, camera_parameters):
#     global_img_path = os.path.join(args.output_dir, 'global', f'rgb_{i:06d}.png')
#     global_images[i] = global_img_path
#     if os.path.exists(global_img_path) and not args.overwrite:
#         return True
#     agent_poses = []
#     for name in names_order:
#         step_data = json.load(open(os.path.join(args.output_dir, 'steps', name, f'{i:06d}.json')))
#         agent_poses.append(step_data["obs"]["pose"])
#     global_image_copy = global_image.copy()
#     global_image_with_agents = render_topdown_locators(global_image_copy, [np.array(agent_pose[:3]) for agent_pose in agent_poses], agent_locator_colors, circle_radii=[15 for _ in agent_poses], camera_parameters=camera_parameters)
#     Image.fromarray(global_image_with_agents).save(global_img_path)
#     return True

def make_image_clip(idx, frame_image_paths):
    return ImageClip(frame_image_paths[idx], duration=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default='output')
    parser.add_argument("--scene", type=str, default='newyork')
    parser.add_argument("--config", type=str, default='agents_num_5')
    parser.add_argument("--agent_type", type=str, choices=['ella', 'generative_agent', 'tour_agent', 'ella_local'], default='generative_agent')
    parser.add_argument("--data_dir", "-d", type=str)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--steps", type=int)
    parser.add_argument("--type", choices=['ego', 'tp'], default='ego')
    parser.add_argument("--codec", choices=['mpeg4', 'h264'], default='h264')
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--skip_merge", action='store_true')

    args = parser.parse_args()
    if args.data_dir is not None:
        args.data_dir = args.data_dir.rstrip('/')
        args.agent_type = args.data_dir.split('/')[-1]
        args.scene = args.data_dir.split('/')[-2].split('_')[0]
        args.output_dir = args.data_dir
    else:
        args.output_dir = os.path.join(args.output_dir, f"{args.scene}_{args.config}", f"{args.agent_type}")
    os.makedirs(os.path.join(args.output_dir, 'global'), exist_ok=True)

    config_path = os.path.join(args.output_dir, 'curr_sim', "config.json")
    with open(config_path, 'r') as file:
        config = json.load(file)

    names_order = config["agent_names"] # by default, can change later
    num_agents = config["num_agents"]
    num_steps = config["step"]

    agent_locator_colors = map_lang_colors_to_rgb(config["locator_colors"])

    config["locator_colors_rgb"] = agent_locator_colors
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)
    print(config_path)

    if args.steps:
        num_steps = min(num_steps, args.steps)

    tpego_images = defaultdict(dict)

    global_images = {}
    global_image_path = os.path.join('/', 'ViCo', 'assets', 'scenes', args.scene, "global.png")
    # global_image = cv2.imread(global_image_path)
    
    if not args.skip_merge:
        steps_folder_path = os.path.join(args.output_dir, "steps")
        merge_step_files(steps_folder_path, names_order, num_steps, overwrite=args.overwrite)

    # camera_parameters = json.load(open(os.path.join('ViCo', 'assets', 'scenes', args.scene, "global_cam_parameters.json")))

    # for name in names_order:
    #     lst_path = None
    #     for frame_idx in range(num_steps):
    #         tpego_img_path = os.path.join(args.output_dir, args.type, name, f'rgb_{frame_idx:06d}.png')
    #         if os.path.exists(tpego_img_path):
    #             lst_path = tpego_img_path
    #         tpego_images[name][frame_idx] = lst_path

    # with ThreadPoolExecutor(max_workers=args.threads) as executor:
    #     global_image_generated = partial(make_global_image, args=args, names_order=names_order, camera_parameters=camera_parameters)
    #     dummy = list(tqdm(executor.map(global_image_generated, range(0, num_steps, args.fps)), total=num_steps//args.fps))

    # with ThreadPoolExecutor(max_workers=args.threads) as executor:
    #     process_frame_partial = partial(make_image_clip, frame_image_paths=global_images)
    #     clips = list(tqdm(executor.map(process_frame_partial, range(0, num_steps, args.fps)), total=num_steps//args.fps))
    #     final_clip = concatenate_videoclips(clips)
    #     final_clip.write_videofile(os.path.join(args.output_dir, 'global', "video.mp4"), fps=args.fps)

    # for name in tqdm(names_order):
    #     with ThreadPoolExecutor(max_workers=args.threads) as executor:
    #         process_frame_partial = partial(make_image_clip, frame_image_paths=tpego_images[name])
    #         clips = list(tqdm(executor.map(process_frame_partial, range(0, num_steps, args.fps)), total=num_steps//args.fps))
    #         final_clip = concatenate_videoclips(clips)
    #         final_clip.write_videofile(os.path.join(args.output_dir, args.type, name, "video.mp4"), fps=args.fps)

    curr_sim_folder_path = os.path.join('/' + args.output_dir, "curr_sim")
    config_path = os.path.join('/' + args.output_dir, "curr_sim/config.json")
    tpego_folder_path = os.path.join('/' + args.output_dir, args.type)
    steps_folder_path = os.path.join('/' + args.output_dir, "steps")
    global_camera_parameters_path = os.path.join('/ViCo/assets/scenes', args.scene, "global_cam_parameters.json")

    if not os.path.exists(os.path.join(args.output_dir, args.type, 'ego_check_exists_lookup.json')):
        ego_check_exists_lookup = defaultdict(list)
        for name in tqdm(names_order):
            for i in tqdm(range(num_steps - 1)):
                if os.path.exists(os.path.join(args.output_dir, args.type, name, f'rgb_{i:06d}.png')):
                    ego_check_exists_lookup[name].append(i)
        with open(os.path.join(args.output_dir, args.type, 'ego_check_exists_lookup.json'), 'w') as file:
            json.dump(ego_check_exists_lookup, file, separators=(',', ':'))

    with open("mytools/demo.html", 'r') as file:
        html_content = file.read()

    html_content = html_content.replace("$OutputFolderPath$", '/' + args.output_dir)
    html_content = html_content.replace("$CurrSimFolderPath$", curr_sim_folder_path)
    html_content = html_content.replace("$ConfigPath$", config_path)
    html_content = html_content.replace("$TpegoFolderPath$", tpego_folder_path)
    html_content = html_content.replace("$StepsFolderPath$", steps_folder_path)
    html_content = html_content.replace("$AvatarImgsPath$", "/ViCo/assets/imgs/avatars")
    html_content = html_content.replace("$GlobalCameraParameterPath$", global_camera_parameters_path)
    html_content = html_content.replace("$GlobalImagePath$", global_image_path)
    html_content = html_content.replace("$FPS$", str(args.fps))

    # with open(os.path.join(args.output_dir, f"demo.html"), 'w') as file:
    #     file.write(html_content)

    url = f"http://localhost:8000/{args.output_dir}/demo.html"
    print(f"Start the server at the Ella root directory (python3 -m http.server) and access the HTML Demo at {url}")
    print(f"For remote access, use the the following command: ssh -L 8080:localhost:8000 -i ~/.ssh/id_rsa username@server_ip_address")
    print("Debugging Tips: you don't see the website is up? Make sure that when you run python mytools/make_html_demo.py -d output/new/DETROIT_agents_num_15/ella, the folder path is relative (i.e. from output), not absolute path from the work directory.")
    print("Also, do not directly navigate to the website html at local, try http://localhost:8000/ first to see if the server is up.")