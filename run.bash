python viz_video.py --dataset_paths \
    /pretrain_data/oxe_lerobot/berkeley_autolab_ur5 \
    /pretrain_data/oxe_lerobot/utaustin_mutex \
    --bbox_jsonl_path /pretrain_data/annotation/v2/bbox/ \
    --sub_task_jsonl_path /pretrain_data/annotation/v2/sub_task/ \
    --save_dir tmp/ \
    --font_path Ubuntu-Regular.ttf \
    --show_object_name
    # --show_plans