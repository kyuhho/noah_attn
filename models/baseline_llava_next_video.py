import argparse
import json
import math
import os
from typing import Optional

import numpy as np
import torch
from decord import VideoReader, cpu
from logzero import logger
from transformers import AutoConfig

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


DEFAULT_QUESTION = "What happens in this video? Please describe it in detail."

class BaselineLLaVANeXTVideo:

    def __init__(
        self,
        model_path: str,
        model_base: Optional[str] = None,
        vision_tower_path: Optional[str] = None,
        conv_mode: str = "vicuna_v1",
        mm_spatial_pool_stride: int = 4,
        mm_spatial_pool_mode: str = "average",
        overwrite: bool = True,
        for_get_frames_num: int = 4,
        load_8bit: bool = False,
        mm_newline_position: str = "no_token",
        force_sample: bool = False,
        add_time_instruction: bool = False,
        output_dir: str = "/output",
    ):
        self.model_path = model_path
        self.model_base = model_base
        self.vision_tower_path = vision_tower_path
        self.conv_mode = conv_mode
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.overwrite = overwrite
        self.for_get_frames_num = for_get_frames_num
        self.load_8bit = load_8bit
        self.mm_newline_position = mm_newline_position
        self.force_sample = force_sample
        self.add_time_instruction = add_time_instruction
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        self._load_model()

    def _load_model(self):
        model_name = get_model_name_from_path(self.model_path)

        if self.overwrite:
            overwrite_config = {
                "mm_spatial_pool_mode": self.mm_spatial_pool_mode,
                "mm_spatial_pool_stride": self.mm_spatial_pool_stride,
                "mm_newline_position": self.mm_newline_position,
            }
            cfg_pretrained = AutoConfig.from_pretrained(self.model_path)

            if "qwen" not in self.model_path.lower():
                mm_vision_tower = getattr(cfg_pretrained, "mm_vision_tower", "") or (self.vision_tower_path or "")
                if "224" in mm_vision_tower:
                    least_token_number = (
                        self.for_get_frames_num * (16 // self.mm_spatial_pool_stride) ** 2 + 1000
                    )
                else:
                    least_token_number = (
                        self.for_get_frames_num * (24 // self.mm_spatial_pool_stride) ** 2 + 1000
                    )

                scaling_factor = math.ceil(least_token_number / 4096)
                if scaling_factor >= 2:
                    name_for_check = getattr(cfg_pretrained, "_name_or_path", "") or self.model_path
                    if "vicuna" in name_for_check.lower():
                        overwrite_config["rope_scaling"] = {
                            "factor": float(scaling_factor),
                            "type": "linear",
                        }
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

            if self.vision_tower_path is not None:
                overwrite_config["mm_vision_tower"] = self.vision_tower_path

            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                self.model_path,
                self.model_base,
                model_name,
                load_8bit=self.load_8bit,
                overwrite_config=overwrite_config,
            )
        else:
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                self.model_path,
                self.model_base,
                model_name,
            )

        if getattr(self.model.config, "force_sample", None) is not None:
            self.force_sample = self.model.config.force_sample
        else:
            self.force_sample = False

        if getattr(self.model.config, "add_time_instruction", None) is not None:
            self.add_time_instruction = self.model.config.add_time_instruction
        else:
            self.add_time_instruction = False

        self.model.eval()
        logger.info("Baseline LLaVA-NeXT-Video model loaded successfully.")

    def load_video(self, video_path: str):
        if self.for_get_frames_num == 0:
            return np.zeros((1, 336, 336, 3)), "0.00s", 0.0

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps())

        frame_idx = list(range(0, len(vr), fps))
        frame_time = [i / fps for i in frame_idx]

        if len(frame_idx) > self.for_get_frames_num or self.force_sample:
            uniform_sampled_frames = np.linspace(
                0, total_frame_num - 1, self.for_get_frames_num, dtype=int
            )
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]

        frame_time_str = ",".join([f"{t:.2f}s" for t in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames, frame_time_str, video_time

    def run_inference(
        self,
        video_id: str,
        video_path: str,
        question: str = DEFAULT_QUESTION,
        output_dir: Optional[str] = None,
        save_result: bool = True,
    ):
        if output_dir is None:
            output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            return None

        video, frame_time, video_time = self.load_video(video_path)
        video = (
            self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
            .half()
            .cuda()
        )
        video = [video]

        qs = question
        if self.add_time_instruction:
            time_instr = (
                f"The video lasts for {video_time:.2f} seconds, and "
                f"{len(video[0])} frames are uniformly sampled from it. "
                f"These frames are located at {frame_time}. "
                f"Please answer the following questions related to this video."
            )
            qs = f"{time_instr}\n{qs}"

        if self.model.config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
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
        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower():
                self.tokenizer.pad_token_id = 151643

        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )

        with torch.inference_mode():
            generate_kwargs = {
                "inputs": input_ids,
                "images": video,
                "attention_mask": attention_masks,
                "modalities": "video",
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": 1024,
                "top_p": 0.1,
                "num_beams": 1,
                "use_cache": True,
                "stopping_criteria": [stopping_criteria],
            }
            output = self.model.generate(**generate_kwargs)

        generated_text = self.tokenizer.batch_decode(
            output, skip_special_tokens=True
        )[0].strip()
        if generated_text.endswith(stop_str):
            generated_text = generated_text[: -len(stop_str)].strip()

        logger.info(f"Question: {question}")
        logger.info(f"Answer:   {generated_text}")

        result = {
            "video_id": video_id,
            "video_path": video_path,
            "question": question,
            "answer": generated_text,
        }

        if save_result:
            save_path = os.path.join(output_dir, f"{video_id}_result.json")
            with open(save_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Result saved → {save_path}")

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=str,
        default="/path/to/video.mp4",
    )
    parser.add_argument("--video_id", type=str, default="result")
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION)
    args = parser.parse_args()

    baseline = BaselineLLaVANeXTVideo(
        model_path="lmms-lab/LLaVA-NeXT-Video-7B",
        model_base=None,
        conv_mode="vicuna_v1",
        mm_spatial_pool_stride=4,
        mm_spatial_pool_mode="average",
        overwrite=True,
        for_get_frames_num=4,
    )

    out = baseline.run_inference(
        video_id=args.video_id,
        video_path=args.video_path,
        question=args.question,
    )
    print(out["answer"])

