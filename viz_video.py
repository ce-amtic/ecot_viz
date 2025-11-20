"""
python ecot/postprocess/viz_annotation.py
"""

import random
import time
import argparse
from pathlib import Path
from PIL import ImageFont, Image
from utils import FastLerobotVLReader, load_jsonlines, annotate_frames, images_to_video

# ===== camera configurations =====
EGO_CAMERA_MAP = {
    # 'dataset name': 'ego camera key name',
    'AgiBotAlpha_covt2lerobot_0818': 'observation.images.cam_2',
    'AgiBotBeta_covt2lerobot_0818': 'observation.images.cam_2',

    'austin_buds_dataset_converted_externally_to_rlds': 'observation.images.cam1',
    'austin_sailor_dataset_converted_externally_to_rlds': 'observation.images.cam1',
    'austin_sirius_dataset_converted_externally_to_rlds': 'observation.images.cam1',
    'stanford_hydra_dataset_converted_externally_to_rlds': 'observation.images.cam1',
    'iamlab_cmu_pickup_insert_converted_externally_to_rlds': 'observation.images.cam1',
    'nyu_franka_play_dataset_converted_externally_to_rlds': 'observation.images.cam',
    'robo_set_new': 'observation.images.cam3',
    'fmb': 'observation.images.cam3',
    'berkeley_autolab_ur5': 'observation.images.cam1',
    'berkeley_fanuc_manipulation': 'observation.images.cam1',
    'qut_dexterous_manpulation': 'observation.images.cam1',
    'taco_play': 'observation.images.cam1',
    'jaco_play': 'observation.images.cam1',
    'utaustin_mutex': 'observation.images.cam1',
    'plex_robosuite': 'observation.images.cam1',
}
THIRD_CAMERA_MAP = {
    # 'dataset name': 'third camera key name',
    'nyu_franka_play_dataset_converted_externally_to_rlds': 'observation.images.cam1',
}

# ===== video visualization =====
def handle_episode(
    reader: FastLerobotVLReader,
    episode_name: str, 
    bbox_jsonl_path: Path | str, 
    sub_task_jsonl_path: Path | str,
    save_dir: Path | str, 
    **kwargs # fps, show_object_name, show_plan, etc.
):  
    """
    给出一个dataloader, 指定的episode_name, 以及bbox和sub_task的jsonl路径,
    生成对应的视频并保存到save_dir中.
    其他参数可选, 包括fps, show_object_name, show_plan等.
    """
    start_time = time.time()
    box_jl = load_jsonlines(bbox_jsonl_path)
    key2bboxes = {(item['frame_id'], item['camera_key']): item['bboxes'] for item in box_jl}

    sub_task_jl = load_jsonlines(sub_task_jsonl_path)
    key2sub_task = {item['frame_id']: item['sub_task'] for item in sub_task_jl}
    key2task = {item['frame_id']: item['task'] for item in sub_task_jl}

    font = None
    font_path = kwargs.get('font_path', None)
    if font_path is not None:
        font = ImageFont.truetype(font_path, 20)

    ts, te = reader.get_episode_range(episode_name)
    tasks = []
    sub_tasks = []
    for t in range(ts, te):
        data = reader[t]
        frame_id = int(data['frame_index'])
        task = key2task.get(frame_id, "")
        tasks.append(task if task else "[no specified task]")
        sub_task = key2sub_task.get(frame_id, "")
        sub_tasks.append(sub_task if sub_task else "[no specified sub-task]")
    third_frames = None
    other_frames = []
    for camera_key in reader.loaded_camera_keys:
        cur_frames = []
        cur_bbox_list = []
        for t in range(ts, te):
            data = reader[t]
            frame_id = int(data['frame_index'])
            cur_frames.append(data[camera_key])
            cur_bbox_list.append(key2bboxes.get((frame_id, camera_key), []))
        if camera_key == reader.third_camera_key:
            cur_frames = annotate_frames(cur_frames, cur_bbox_list, tasks, sub_tasks, show_plans=kwargs.get('show_plans', False), show_object_name=kwargs.get('show_object_name', True), font=font)
            third_frames = cur_frames
        else:
            cur_frames = annotate_frames(cur_frames, cur_bbox_list, show_plans=kwargs.get('show_plans', False), show_object_name=kwargs.get('show_object_name', True), font=font)
            other_frames.append(cur_frames)
    frames = []
    sizes = [third_frames[0].size] + [f[0].size for f in other_frames]
    total_width = sum([s[0] for s in sizes])
    max_height = max([s[1] for s in sizes])
    for t in range(te - ts):
        new_frame = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for f in [third_frames] + other_frames:
            new_frame.paste(f[t], (x_offset, 0))
            x_offset += f[t].size[0]
        frames.append(new_frame)

    print(f"Annotated {len(frames)} frames for episode {episode_name} in {time.time() - start_time:.2f}s.")
    save_path = Path(save_dir) / reader.name / f"{episode_name}.mp4"
    images_to_video(
        images=frames,
        output_path=save_path,
        fps=kwargs['fps'],
    )
    
def handle_dataset(
    dataset_path: str | Path,
    bbox_jsonl_path: str | Path,
    sub_task_jsonl_path: str | Path,
    save_dir: str | Path,
    episodes: list[str] | None = None,
    **kwargs # fps, show_object_name, show_plan, etc.
):
    """
    给出一个dataset_path, 以及bbox和sub_task的jsonl路径,
    生成对应的数据集中的所有episode和camera_key的视频并保存到save_dir中.
    如果指定了episodes, 则只处理这些episode.
    其他参数可选, 包括fps, show_object_name, show_plan等.
    """
    dataset_name = Path(dataset_path).name
    dataset = FastLerobotVLReader(
        root=dataset_path,
        ego_name=EGO_CAMERA_MAP.get(dataset_name, None),
        third_name=THIRD_CAMERA_MAP.get(dataset_name, None),
        image_resize=(768, -1),
        prefetch_num=0,
    )
    dataset.ego_camera_key = None
    dataset.loaded_camera_keys = [dataset.third_camera_key]

    if episodes is None:
        episodes = dataset.all_episode_stems
    random.shuffle(episodes)

    for ep in episodes:
        print(f"Processing {dataset.name} - {ep}")
        handle_episode(
            reader=dataset,
            episode_name=ep,
            bbox_jsonl_path=Path(bbox_jsonl_path) / dataset.name / f"{ep}.bbox.jsonl",
            sub_task_jsonl_path=Path(sub_task_jsonl_path) / dataset.name / f"{ep}.sub_task.jsonl",
            save_dir=Path(save_dir),
            fps=dataset.meta.fps,
            **kwargs
        )

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_paths", type=str, nargs='+', required=True, help="List of dataset paths to process.")
    argparser.add_argument("--eps", type=str, nargs='*', default=None, help="List of episode names to process. If not provided, all episodes will be processed.")
    argparser.add_argument("--bbox_jsonl_path", type=str, required=True, help="Path to the bbox jsonl annotations.")
    argparser.add_argument("--sub_task_jsonl_path", type=str, required=True, help="Path to the sub-task jsonl annotations.")
    argparser.add_argument("--save_dir", type=str, required=True, help="Directory to save the output videos.")
    argparser.add_argument("--font_path", type=str, required=True, help="Path to the font file for annotations.")
    argparser.add_argument("--show_object_name", action='store_true', help="Whether to show object names in the annotations.")
    argparser.add_argument("--show_plans", action='store_true', help="Whether to show plans in the annotations.")
    args = argparser.parse_args()

    for dataset_path in args.dataset_paths:
        handle_dataset(
            dataset_path=dataset_path,
            bbox_jsonl_path=args.bbox_jsonl_path,
            sub_task_jsonl_path=args.sub_task_jsonl_path,
            save_dir=args.save_dir,
            episodes=args.eps,
            show_object_name=args.show_object_name,
            show_plans=args.show_plans,
            font_path=args.font_path,
        )