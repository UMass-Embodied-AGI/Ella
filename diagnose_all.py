# diagnose_all.py
import os
import sys
import time
import argparse
import numpy as np
import json
import math
import genesis as gs

from vico.env import VicoEnv
from vico.tools.utils import atomic_save, json_converter
from vico.tools.constants import ASSETS_PATH
from agents import AgentProcess, get_agent_cls
from PIL import Image 
from tools.model_manager import global_model_manager

global_model_manager.init(local=True)

DEBUG_OUTDIR = "diagnose_debug_detroit"
os.makedirs(DEBUG_OUTDIR, exist_ok=True)

def _get_pose_tuple(env, idx):
    try:
        pose = env.config["agent_poses"][idx]
    except Exception:
        try:
            pose = env.agent_poses[idx]
        except Exception:
            pose = env.obs[idx].get('pose', None)
            if pose is None:
                raise RuntimeError("Unable to fetch agent pose from env.")
    return pose  

CAM_CONV = {} 

def body_xyz(env, idx):
    return env.agents[idx].robot.global_trans.copy()

def _cam_world_from_extr(E, mode):
    E = np.asarray(E, np.float32)
    if mode == "c2w":
        return E[:3, 3].copy()
    R, t = E[:3,:3], E[:3,3]
    return (-R.T @ t).astype(np.float32)

def _fwd_world_from_extr(E, mode):
    E = np.asarray(E, np.float32)
    R = E[:3, :3]
    if mode == "c2w":
        fwd = -R[:, 2]
    else:  
        fwd = -(R.T)[:, 2]
    fwd /= (np.linalg.norm(fwd) + 1e-8)
    return fwd

def live_cam_xyz(env, idx):
    E = env.obs[idx]["extrinsics"]
    mode = CAM_CONV.get(idx, "c2w")
    return _cam_world_from_extr(E, mode)

def live_cam_fwd(env, idx):
    E = env.obs[idx]["extrinsics"]
    mode = CAM_CONV.get(idx, "c2w")
    return _fwd_world_from_extr(E, mode)

def calibrate_cam_mode(env, idx, world_target_xyz):
    env.get_obs([idx])
    E = env.obs[idx]["extrinsics"]
    c2w = E[:3,3]
    w2c = -E[:3,:3].T @ E[:3,3]
    err_c2w = np.linalg.norm(c2w - world_target_xyz)
    err_w2c = np.linalg.norm(w2c - world_target_xyz)
    if err_c2w > 100: 
        CAM_CONV[idx] = "c2w" 
    else:
        CAM_CONV[idx] = "c2w" 

def live_cam_xyz(env, idx):
    E = env.obs[idx]["extrinsics"]
    mode = CAM_CONV.get(idx, "c2w")
    return _cam_world_from_extr(E, mode)


def dump_pair_debug(env, a_idx, b_idx, a_obs, tag="", anchor_index=0):
    anchor_index = str(anchor_index)
    env.get_obs([a_idx, b_idx])  
    a_pos = live_cam_xyz(env, a_idx)       
    b_pos = live_cam_xyz(env, b_idx)       
    a_fwd = live_cam_fwd(env, a_idx)   

    dxy = np.array([b_pos[0] - a_pos[0], b_pos[1] - a_pos[1]], np.float32)
    dist = float(np.linalg.norm(dxy))
    bearing_deg = math.degrees(math.atan2(dxy[1], dxy[0]))
    a_yaw_deg   = math.degrees(math.atan2(a_fwd[1], a_fwd[0]))
    delta_deg   = ((bearing_deg - a_yaw_deg + 180) % 360) - 180

    rgb = a_obs['rgb']
    depth = a_obs['depth']
    fov = a_obs.get('fov', None)
    extr = a_obs.get('extrinsics', None)
    cur_place = a_obs.get('current_place', None)

    stem = f"A{a_idx}_sees_B{b_idx}"
    os.makedirs(os.path.join(DEBUG_OUTDIR, anchor_index), exist_ok=True)
    rgb_path   = os.path.join(DEBUG_OUTDIR, anchor_index, stem + "_rgb.png")
    depth_path = os.path.join(DEBUG_OUTDIR, anchor_index, stem + "_depth.png")
    meta_path  = os.path.join(DEBUG_OUTDIR, anchor_index, stem + "_cam.json")

    Image.fromarray(rgb).save(rgb_path)

    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid.any():
        dmin, dmax = float(d[valid].min()), float(d[valid].max())
        dviz = (np.clip((d - dmin) / (dmax - dmin), 0, 1) * 255.0).astype(np.uint8) if dmax > dmin else np.zeros_like(d, np.uint8)
    else:
        dviz = np.zeros_like(d, np.uint8)
    Image.fromarray(dviz).save(depth_path)

    meta = {
        "asker_idx": a_idx,
        "target_idx": b_idx,
        "A_cam_mode": CAM_CONV.get(a_idx),
        "B_cam_mode": CAM_CONV.get(b_idx),
        "A_cam_pos": a_pos.tolist(),
        "B_cam_pos": b_pos.tolist(),
        "A_cam_fwd": a_fwd.tolist(),
        "distance_xy_m": dist,
        "bearing_deg": bearing_deg,
        "delta_yaw_deg": delta_deg,
        "fov": fov,
        "current_place": cur_place,
        "extrinsics": extr,
        "anchor_index": anchor_index
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=lambda x: np.asarray(x).tolist())

    print(f"Saved A-view RGB → {rgb_path}")
    print(f"Saved A-view depth → {depth_path}")
    print(f"Saved camera meta → {meta_path}")

def put_b_in_front_of_a(env, a_idx, b_idx, dist=3.5, a_xyz=(60.0, 330.0, 119.3399586), place="open space", park_others=True):
    max_tries = 4 

    for idx in (a_idx, b_idx):
        env.perform_action(idx, {"type": "enter", "arg1": place})
    env.scene_step()

    ax, ay, az = map(float, a_xyz)
    
    for t in range(max_tries):
        env.perform_action(a_idx, {"type": "teleport",
                                "arg1": np.array([ax, ay, az], dtype=np.float32)})

        env.scene_step()
        env.get_obs([a_idx]) 

    bx, by, bz = ax + float(dist), ay, az
    print(f"cam pos before teleop, {env.agents[b_idx].ego_view.pos}")
    for t in range(max_tries):
        env.perform_action(b_idx, {"type": "teleport",
                               "arg1": np.array([bx, by, bz], dtype=np.float32)})

        env.scene_step()
        env.get_obs([b_idx])
    print(f"cam pos before teleop, {env.agents[b_idx].ego_view.pos}")
    print("sanity check for bx, by, bz: ", bx, by, bz)

    Eb_check = np.asarray(env.obs[b_idx]["extrinsics"], np.float32)
    print("After teleporting: ", Eb_check[:3,3])
    print("After teleporting with body pose: ", body_xyz(env, b_idx))

    a_target = np.array([ax, ay, az], dtype=np.float32)
    b_target = np.array([bx, by, bz], dtype=np.float32)

    env.perform_action(a_idx, {"type": "look_at",
                               "arg1": body_xyz(env, b_idx)})
    env.perform_action(b_idx, {"type": "look_at",
                               "arg1": body_xyz(env, a_idx)})
    env.scene_step()


    env.get_obs([a_idx, b_idx])
    a_obs, b_obs = env.obs[a_idx], env.obs[b_idx]

    Ea = np.asarray(a_obs["extrinsics"], dtype=np.float32)
    Eb = np.asarray(b_obs["extrinsics"], dtype=np.float32)
    
    a_live = live_cam_xyz(env, a_idx)
    b_live = live_cam_xyz(env, b_idx)
    
    E = np.asarray(env.obs[b_idx]["extrinsics"], np.float32)
    b_target = np.array([ax + float(dist), ay, az], np.float32)

    
    pos_c2w = E[:3, 3]                                   
    pos_w2c = (-E[:3, :3].T @ E[:3, 3]).astype(np.float32) 

    err_c2w = float(np.linalg.norm(pos_c2w - b_target))
    err_w2c = float(np.linalg.norm(pos_w2c - b_target))
    print(f"[calib] B{b_idx} err_c2w={err_c2w:.3f}  err_w2c={err_w2c:.3f}")
    
    debug = {
        "A_xyz": a_live.tolist(),
        "B_xyz": b_live.tolist(),
        "dist_xy_m": float(np.linalg.norm(a_live[:2] - b_live[:2])),
        "place": a_obs.get("current_place")
    }

    return a_obs, b_obs

def face_each_other_true(env, a_idx: int, b_idx: int, dist: float = 3.5,
                         angle_tol_deg: float = 3.0, max_iters: int = 16):
    """Rotate bodies until A and B face each other, using calibrated camera frames."""

    def angdiff(a_deg, b_deg):
        return (a_deg - b_deg + 180.0) % 360.0 - 180.0

    def yaw_from_vec(v):  
        return float(np.degrees(np.arctan2(v[1], v[0])))

    env.get_obs([a_idx, b_idx])

    for _ in range(max_iters):
        Apos = live_cam_xyz(env, a_idx)
        Bpos = live_cam_xyz(env, b_idx)
        Afwd = live_cam_fwd(env, a_idx)
        Bfwd = live_cam_fwd(env, b_idx)

        
        des_yaw_a = yaw_from_vec(Bpos - Apos)
        des_yaw_b = yaw_from_vec(Apos - Bpos)

        cur_yaw_a = yaw_from_vec(Afwd)
        cur_yaw_b = yaw_from_vec(Bfwd)

        dA = angdiff(des_yaw_a, cur_yaw_a)
        dB = angdiff(des_yaw_b, cur_yaw_b)
        print(f"[DEBUG] headings (deg)  A: cur={cur_yaw_a:.1f}→{des_yaw_a:.1f} (Δ={dA:.1f}) | "
              f"B: cur={cur_yaw_b:.1f}→{des_yaw_b:.1f} (Δ={dB:.1f})")

        if abs(dA) <= angle_tol_deg and abs(dB) <= angle_tol_deg:
            break

        if abs(dA) > angle_tol_deg:
            env.perform_action(a_idx, {
                "type": "turn_left" if dA > 0 else "turn_right",
                "arg1": float(min(abs(dA), 30.0))
            })
        if abs(dB) > angle_tol_deg:
            env.perform_action(b_idx, {
                "type": "turn_left" if dB > 0 else "turn_right",
                "arg1": float(min(abs(dB), 30.0))
            })

        env.scene_step()
        env.get_obs([a_idx, b_idx])

    Apos = live_cam_xyz(env, a_idx)
    Bpos = live_cam_xyz(env, b_idx)
    env.perform_action(a_idx, {"type": "look_at", "arg1": Bpos.astype(np.float32)})
    env.perform_action(b_idx, {"type": "look_at", "arg1": Apos.astype(np.float32)})
    env.scene_step()
    env.get_obs([a_idx, b_idx])

    final_b_target = np.array([Apos[0] + dist, Apos[1], Apos[2]], np.float32)

def log_line(env, a_idx, b_idx):
    env.get_obs([a_idx, b_idx])
    Ea = np.asarray(env.obs[a_idx]["extrinsics"], np.float32); Eb = np.asarray(env.obs[b_idx]["extrinsics"], np.float32)
    A = Ea[:3, 3]; B = Eb[:3, 3]; fwd = -Ea[:3, :3][:, 2]; fwd /= (np.linalg.norm(fwd) + 1e-8)
    print(f"A{a_idx}[{A[0]:.2f},{A[1]:.2f},{A[2]:.2f}] look[{fwd[0]:+.3f},{fwd[1]:+.3f},{fwd[2]:+.3f}] | B{b_idx}[{B[0]:.2f},{B[1]:.2f},{B[2]:.2f}]")

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="NY")
    parser.add_argument("--config", type=str, default="agents_num_15")
    parser.add_argument("--output_dir", "-o", type=str, default="output")
    parser.add_argument("--num_agents", type=int, default=15)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--enable_gt_segmentation", action="store_true")  # optional
    parser.add_argument("--skip_avatar_animation", action="store_true")
    parser.add_argument("--agent_type", type=str, default="ella")
    parser.add_argument("--lm_source", type=str, default="azure")
    parser.add_argument("--lm_id", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--external_run", type=str, default="")
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, f"{args.scene}_{args.config}", f"{args.agent_type}")
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, 'curr_sim')
    if not os.path.exists(config_path):
        seed_config_path = os.path.join(ASSETS_PATH, 'scenes', args.scene, args.config)
        print(f"Seed sim from {seed_config_path}")
        import shutil, errno
        try:
            shutil.copytree(seed_config_path, config_path)
        except OSError as exc:
            if exc.errno in (errno.ENOTDIR, errno.EINVAL):
                shutil.copy(seed_config_path, config_path)
            else:
                raise

    env = VicoEnv(
        seed=0,
        precision='32',
        logging_level='info',
        backend=gs.gpu,
        head_less=True,
        resolution=args.resolution,
        challenge='full',
        num_agents=args.num_agents,
        config_path=config_path,
        scene=args.scene,
        enable_indoor_scene=False,
        enable_indoor_objects=False,
        enable_outdoor_objects=False,
        enable_collision=False,
        enable_decompose=False,
        skip_avatar_animation=True,
        enable_gt_segmentation=args.enable_gt_segmentation,
        no_load_scene=False,
        output_dir=args.output_dir,
        enable_third_person_cameras=False,  
        enable_demo_camera=False,
        no_traffic_manager=True,
        enable_tm_debug=False,
        tm_vehicle_num=0,
        tm_avatar_num=0,
        save_per_seconds=99999,  
        defer_chat=True,
        debug=False,
        batch_renderer=False,
    )

    
    agents = []
    agent_cls = get_agent_cls(agent_type=args.agent_type)
    for i in range(args.num_agents):
        # if env.agent_names[i] in exclude_agent_list:
        #     exclude_ids.append(i)

        basic_kwargs = dict(
            name = env.agent_names[i],
            pose = env.config["agent_poses"][i],
            info = env.agent_infos[i],
            sim_path = config_path,
            debug = False,
        )
        llm_kwargs = dict(lm_source=args.lm_source, lm_id=args.lm_id)
        agents.append(agent_cls(**basic_kwargs, **llm_kwargs))

    obs = env.reset()

    # positions_to_test = [(60.0, 330.0, 119.3399586), (10.0, 25.0, 119.3399586), (300.0, 330.0, 119.3399586), (150.0, 150.0, 119.3399586)]
    positions_to_test = [(60.0, 330.0, 119.3399586)]
    agent_idx_to_name = {i: env.agent_names[i] for i in range(args.num_agents)}

    results = []
    for anchor_index, a_anchor in enumerate(positions_to_test):
        for a_idx in range(args.num_agents):

            for b_idx in range(args.num_agents):
                if env.agent_names[b_idx] != 'Naomi Zhang':
                    continue 
                
                if a_idx == b_idx: 
                    continue
 
                env.perform_action(a_idx, {"type": "teleport",
                                    "arg1": np.array(a_anchor, dtype=np.float32)})
                env.scene_step()
                
                put_b_in_front_of_a(env, a_idx, b_idx, a_xyz=a_anchor, dist=2.0)
                face_each_other_true(env, a_idx, b_idx, angle_tol_deg=8.0)

                env.get_obs([a_idx])
                a_obs = env.obs[a_idx]
                rgb = a_obs['rgb']
                depth = a_obs['depth']
                fov = a_obs['fov']
                extr = a_obs['extrinsics']

                env.perform_action(a_idx, {"type":"wake"})
                env.perform_action(a_idx, {"type":"stand"})

                env.perform_action(b_idx, {"type":"wake"})
                env.perform_action(b_idx, {"type":"stand"})

                log_line(env, a_idx, b_idx)

                dump_pair_debug(env, a_idx, b_idx, a_obs, tag=f"A{a_idx}_vs_B{b_idx}", anchor_index=anchor_index)

                out = agents[a_idx].diagnose(
                    text_prompt=f"What is the name of the avatar?",
                    rgb=rgb, depth=depth, 
                    extrinsics=extr,
                    external_run=args.external_run, 
                    agent_name=env.agent_names[a_idx]
                )

                print("asker: ", env.agent_names[a_idx])
                print("target: ", env.agent_names[b_idx])
                print("Answer: ", out)
                print("-"*30)
                
                results.append({
                    "asker": env.agent_names[a_idx],
                    "target": env.agent_names[b_idx],
                    "answer": out,
                    "position": str(anchor_index)
                })
                print(f"[{env.agent_names[a_idx]}] sees [{env.agent_names[b_idx]}] -> {out}")

                
                env.perform_action(b_idx, {"type": "teleport",
                                "arg1": np.array((1e5, 1e5, 1e5), dtype=np.float32)})
                env.scene_step()
             
            
            env.perform_action(a_idx, {"type": "teleport",
                                "arg1": np.array((1e5, 1e5, 1e5), dtype=np.float32)})
            env.scene_step()

        env.perform_action(a_idx, {"type": "teleport",
                                "arg1": np.array((1e5, 1e5, 1e5), dtype=np.float32)})
        env.scene_step()

    report_path = os.path.join(args.output_dir, "diagnose_all_vs_all_llm_newyork.json")
    atomic_save(report_path, json.dumps(results, indent=2, default=json_converter))
    print(f"Saved report to {report_path}")

if __name__ == "__main__":
    run()