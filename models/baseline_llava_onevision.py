import copy
import json
import math
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from logzero import logger
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token

warnings.filterwarnings("ignore")

DEFAULT_QUESTION = "What happens in this video? Please describe it in detail."

class BaselineLLaVAOneVision:
    def __init__(
        self,
        model_path,
        model_base=None,
        vision_tower_path=None,
        conv_mode="qwen_1_5",
        for_get_frames_num=16,
        device_map="auto",
        output_dir="/output/llava_onevision/baseline",
    ):
        self.model_path = model_path
        self.model_base = model_base
        self.vision_tower_path = vision_tower_path
        self.conv_mode = conv_mode
        self.for_get_frames_num = for_get_frames_num
        self.device_map = device_map
        self.output_dir = output_dir

        self._load_model()

    def _load_model(self):
        model_name = "llava_qwen"
        llava_model_args = {"multimodal": True}
        if self.vision_tower_path is not None:
            llava_model_args["overwrite_config"] = {"mm_vision_tower": self.vision_tower_path}

        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            self.model_path,
            self.model_base,
            model_name,
            device_map=self.device_map,
            attn_implementation="eager",
            **llava_model_args,
        )
        self.model.eval()

    def load_video(self, video_path):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, self.for_get_frames_num, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames

    def run_inference(
        self,
        video_id,
        video_path,
        question="Describe what's happening in this video.",
        max_new_tokens=4096,
        output_dir=None,
        save_result=False,
    ):
        if output_dir is None:
            output_dir = self.output_dir

        video_frames = self.load_video(video_path)
        image_tensors = []
        frames = (
            self.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
            .half()
            .cuda()
        )
        image_tensors.append(frames)

        conv = copy.deepcopy(conv_templates[self.conv_mode])
        qs = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        
        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=None,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
                modalities=["video"],
            )
        

        generated_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)[
            0
        ].strip()

        result = {
            "video_id": video_id,
            "video_path": video_path,
            "question": question,
            "answer": generated_text,
        }

        if save_result:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{video_id}_result.json")
            with open(save_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="/path/to/video.mp4")
    parser.add_argument("--video_id", type=str, default="demo")
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION)
    args = parser.parse_args()

    manipulator = BaselineLLaVAOneVision(
        model_path="lmms-lab/llava-onevision-qwen2-7b-ov",
        model_base=None,
        for_get_frames_num=8,
    )

    result = manipulator.run_inference(
        video_id=args.video_id,
        video_path=args.video_path,
        question=args.question,
    )
    print(result["answer"])
