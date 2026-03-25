import json
import os
import argparse

import numpy as np
from tqdm import tqdm
from models.noah_llava_next_video import NoahLLaVANeXTVideo



def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

DEFAULT_QUESTION = "What happens in this video? Please describe it in detail."

DEFAULT_CONFIG = {
    "event_similarity_threshold": 0.94,
    "cross_event_bias_value": -5.0,
    "cross_event_layers": [0, 1],
    "features": ["vision_focus_dominant"],
    "vision_focus_boost": 1.0,
    "vision_focus_layers": [[8, 11]],
    "vision_focus_sparse_threshold": 0.3,
    "vision_focus_confident_threshold": 0.4,
}

OUT_PATH = "/home/work/Redteaming/kyuho/noah/output/results/llava_next_video/captioning/results_vfd_1_0_layer_8_11.json"

def main():
    data = load_json('/home/work/Redteaming/kyuho/noah/metadata_dev.json')

    inferencer = NoahLLaVANeXTVideo(
        model_path="/home/work/Redteaming/data1/noah/LLaVA-NeXT-Video-7B",
        model_base=None,
        vision_tower_path="/home/work/Redteaming/data1/noah/clip-vit-large-patch14-336",
        conv_mode="vicuna_v1",
        mm_spatial_pool_stride=4,
        mm_spatial_pool_mode="average",
        overwrite=True,
        for_get_frames_num=8,
        output_dir=f"/home/work/Redteaming/kyuho/noah/output/results",
    )

    results = []
    
    for item in tqdm(data):
        video_id = item["id"]
        video_dir = "/home/work/Redteaming/kyuho/noah/dataset/noah"
        video_path = os.path.join(video_dir, video_id + ".mp4")

        result = inferencer.run_inference(
            video_id=video_id,
            video_path=video_path,
            question=DEFAULT_QUESTION,
            event_similarity_threshold=DEFAULT_CONFIG["event_similarity_threshold"],
            event_bias_value=DEFAULT_CONFIG["cross_event_bias_value"],
            cross_event_layers=DEFAULT_CONFIG["cross_event_layers"],
            features=DEFAULT_CONFIG["features"],
            vision_focus_boost=DEFAULT_CONFIG["vision_focus_boost"],
            vision_focus_layers=DEFAULT_CONFIG["vision_focus_layers"],
            vision_focus_sparse_threshold=DEFAULT_CONFIG["vision_focus_sparse_threshold"],
            vision_focus_confident_threshold=DEFAULT_CONFIG["vision_focus_confident_threshold"],
            save_result=False,
        )

        results.append({
            "id": video_id,
            "caption": result["answer"],
            "status": "success",
        })

        save_json(results, OUT_PATH)

    print(f"Saved result to {OUT_PATH}")


if __name__ == "__main__":
    main()