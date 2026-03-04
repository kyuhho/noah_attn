import torch
import torch.nn.functional as F
import json
import os
import math
import numpy as np
import argparse

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model.language_model.modeling_llama import apply_rotary_pos_emb, repeat_kv
from decord import VideoReader, cpu
from transformers import AutoConfig
from logzero import logger

DEFAULT_CONFIG = {
    "event_similarity_threshold": 0.94,
    "cross_event_bias_value": -7.0,
    "cross_event_layers": [0, 15],
    "vision_focus_boost": 0.0,
    "vision_focus_layers": [0, 31],
}

DEFAULT_QUESTION = "What happens in this video? Please describe it in detail."

class NoahLLaVANeXTVideo:
    def __init__(
        self,
        model_path,
        model_base=None,
        vision_tower_path=None,
        conv_mode="vicuna_v1",
        mm_spatial_pool_stride=4,
        mm_spatial_pool_mode="average",
        overwrite=True,
        for_get_frames_num=8,
        load_8bit=False,
        mm_newline_position="no_token",
        force_sample=False,
        add_time_instruction=False,
        output_dir="/output",
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
        
        self._patched_forwards: dict = {}
        self._token_info: dict | None = None
        self._events: list = []
        self._event_bias_value: float = 0.0
        self._vision_focus_boost: float = 0.0
        self._features: set = set()
        self._dominant_event_idx: int | None = None

        os.makedirs(self.output_dir, exist_ok=True)
        self._load_model()

    def _load_model(self):
        model_name = get_model_name_from_path(self.model_path)
        cfg_pretrained = AutoConfig.from_pretrained(self.model_path)

        if self.overwrite:
            overwrite_config = {
                "mm_spatial_pool_mode": self.mm_spatial_pool_mode,
                "mm_spatial_pool_stride": self.mm_spatial_pool_stride,
                "mm_newline_position": self.mm_newline_position,
            }
            if self.vision_tower_path is not None:
                overwrite_config["mm_vision_tower"] = self.vision_tower_path

            mm_vision_tower = (
                getattr(cfg_pretrained, "mm_vision_tower", "")
                or (self.vision_tower_path or "")
            )
            if "qwen" not in self.model_path.lower():
                if "224" in mm_vision_tower:
                    least_token_number = (
                        self.for_get_frames_num
                        * (16 // self.mm_spatial_pool_stride) ** 2
                        + 1000
                    )
                else:
                    least_token_number = (
                        self.for_get_frames_num
                        * (24 // self.mm_spatial_pool_stride) ** 2
                        + 1000
                    )

                scaling_factor = math.ceil(least_token_number / 4096)
                if scaling_factor >= 2:
                    _name = (
                        getattr(cfg_pretrained, "_name_or_path", "")
                        or self.model_path
                    )
                    if "vicuna" in _name.lower():
                        overwrite_config["rope_scaling"] = {
                            "factor": float(scaling_factor),
                            "type": "linear",
                        }
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = (
                        4096 * scaling_factor
                    )

            self.tokenizer, self.model, self.image_processor, _ = (
                load_pretrained_model(
                    self.model_path,
                    self.model_base,
                    model_name,
                    load_8bit=self.load_8bit,
                    overwrite_config=overwrite_config,
                    attn_implementation="eager",
                )
            )
        else:
            self.tokenizer, self.model, self.image_processor, _ = (
                load_pretrained_model(
                    self.model_path,
                    self.model_base,
                    model_name,
                    attn_implementation="eager",
                )
            )

        if getattr(self.model.config, "force_sample", None) is not None:
            self.force_sample = self.model.config.force_sample
        else:
            self.force_sample = False

        if getattr(self.model.config, "add_time_instruction", None) is not None:
            self.add_time_instruction = self.model.config.add_time_instruction
        else:
            self.add_time_instruction = False

        self.cfg_pretrained = cfg_pretrained
        logger.info("Model loaded successfully (eager attention).")

    def load_video(self, video_path):
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

    def _make_patched_forward(self, original_forward, layer_idx):
        """Single patched forward: cross-event (prefill), vision_focus (decode), etc."""
        module = self.model.model.layers[layer_idx].self_attn

        def forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()

            query_states = module.q_proj(hidden_states)
            key_states = module.k_proj(hidden_states)
            value_states = module.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, module.num_heads, module.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)

            past_key_value = getattr(module, "past_key_value", past_key_value)
            cos, sin = module.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, module.layer_idx, cache_kwargs
                )
            key_states = repeat_kv(key_states, module.num_key_value_groups)
            value_states = repeat_kv(value_states, module.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(module.head_dim)
            
            if attention_mask is not None:
                causal_mask = attention_mask
                if cache_position is not None:
                    dim2 = attention_mask.shape[2]
                    if dim2 == 1:
                        # decode: mask has only one row, use it (avoid OOB)
                        causal_mask = attention_mask[:, :, 0:1, : key_states.shape[-2]]
                    else:
                        causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # Prefill: set token_per_frame / visual_token_len when vision_focus_greedy or vision_focus_dominant is on
            if (
                q_len > 1
                and ("vision_focus_greedy" in self._features or "vision_focus_dominant" in self._features)
                and hasattr(self, "_vision_focus_layers")
                and layer_idx in self._vision_focus_layers
            ):
                info = self._token_info
                if info is not None and info.get("token_per_frame") is None:
                    vis_len = q_len - (info["original_ids_len"] - 1)
                    info["token_per_frame"] = vis_len // info["num_frames"]
                    info["visual_token_len"] = vis_len

            # cross-event boundary bias (only when feature enabled)
            if (
                "cross_event_boundary" in self._features
                and q_len > 1
                and hasattr(self, "_cross_event_layers")
                and layer_idx in self._cross_event_layers
            ):
                info = self._token_info
                events = self._events
                bias_value = self._event_bias_value
                kv_len = key_states.shape[-2]
                if info and events and len(events) > 1 and bias_value != 0:
                    if info.get("token_per_frame") is None:
                        vis_len = q_len - (info["original_ids_len"] - 1)
                        info["token_per_frame"] = vis_len // info["num_frames"]
                        info["visual_token_len"] = vis_len
                    vid_start = info["video_index"]
                    tpf = info["token_per_frame"]
                    num_frames = info["num_frames"]
                    if tpf and num_frames and vid_start + num_frames * tpf <= kv_len:
                        bias = torch.zeros(1, 1, q_len, kv_len, device=attn_weights.device, dtype=attn_weights.dtype)
                        # Consecutive events only: first frame of next -> last frame of prev (same as noah_llava_next_video.py)
                        for i in range(len(events) - 1):
                            prev_event, next_event = events[i], events[i + 1]
                            src = next_event[0]
                            tgt = prev_event[-1]
                            if not (1 <= src <= num_frames and 1 <= tgt <= num_frames):
                                continue
                            src_s = vid_start + (src - 1) * tpf
                            src_e = vid_start + src * tpf
                            tgt_s = vid_start + (tgt - 1) * tpf
                            tgt_e = vid_start + tgt * tpf
                            src_s, src_e = max(0, src_s), min(src_e, q_len, kv_len)
                            tgt_s, tgt_e = max(0, tgt_s), min(tgt_e, q_len, kv_len)
                            if src_s < src_e and tgt_s < tgt_e:
                                bias[0, 0, src_s:src_e, tgt_s:tgt_e] += bias_value
                        attn_weights = attn_weights + bias

            # cross-event: later events attending to earlier events (full suppress)
            if (
                "cross_event" in self._features
                and q_len > 1
                and hasattr(self, "_cross_event_layers")
                and layer_idx in self._cross_event_layers
            ):
                info = self._token_info
                events = self._events
                bias_value = self._event_bias_value
                kv_len = key_states.shape[-2]
                if info and events and len(events) > 1 and bias_value != 0:
                    if info.get("token_per_frame") is None:
                        vis_len = q_len - (info["original_ids_len"] - 1)
                        info["token_per_frame"] = vis_len // info["num_frames"]
                        info["visual_token_len"] = vis_len
                    vid_start = info["video_index"]
                    tpf = info["token_per_frame"]
                    num_frames = info["num_frames"]
                    if tpf and num_frames and vid_start + num_frames * tpf <= kv_len:
                        bias = torch.zeros(1, 1, q_len, kv_len, device=attn_weights.device, dtype=attn_weights.dtype)
                        for i in range(len(events)):
                            for j in range(i + 1, len(events)):
                                earlier_event, later_event = events[i], events[j]
                                for src_frame in later_event:
                                    for tgt_frame in earlier_event:
                                        if not (1 <= src_frame <= num_frames and 1 <= tgt_frame <= num_frames):
                                            continue
                                        src_s = vid_start + (src_frame - 1) * tpf
                                        src_e = vid_start + src_frame * tpf
                                        tgt_s = vid_start + (tgt_frame - 1) * tpf
                                        tgt_e = vid_start + tgt_frame * tpf
                                        src_s, src_e = max(0, src_s), min(src_e, q_len, kv_len)
                                        tgt_s, tgt_e = max(0, tgt_s), min(tgt_e, q_len, kv_len)
                                        if src_s < src_e and tgt_s < tgt_e:
                                            bias[0, 0, src_s:src_e, tgt_s:tgt_e] += bias_value
                        attn_weights = attn_weights + bias

            # Vision focus (decode only): probe attention → pick event with most attention → boost its frames
            if (
                ("vision_focus_greedy" in self._features or "vision_focus_dominant" in self._features)
                and q_len == 1
                and hasattr(self, "_vision_focus_layers")
                and layer_idx in self._vision_focus_layers
            ):
                info = self._token_info
                events = self._events
                kv_len = key_states.shape[-2]
                if (
                    info is not None
                    and info.get("visual_token_len") is not None
                    and events
                ):
                    vid_start = info["video_index"]
                    vis_len = info["visual_token_len"]
                    tpf = info["token_per_frame"]
                    num_frames = info["num_frames"]
                    if tpf and num_frames and vid_start + vis_len <= kv_len:
                        first_vf_layer = min(self._vision_focus_layers)
                        use_dominant = "vision_focus_dominant" in self._features

                        if use_dominant and layer_idx != first_vf_layer:
                            # Use event chosen by first layer; no recompute
                            best_idx = self._dominant_event_idx
                        else:
                            # First layer (or greedy): compute best event for this layer
                            if use_dominant and layer_idx == first_vf_layer:
                                self._dominant_event_idx = None
                            probe = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
                            vis_attn = probe[0, :, 0, vid_start : vid_start + vis_len].mean(dim=0)
                            if vis_attn.numel() != num_frames * tpf:
                                best_idx = -1
                            else:
                                per_frame = vis_attn.view(num_frames, tpf).sum(dim=1).detach().cpu()
                                best_idx, best_avg = -1, -1.0
                                for ei, ev in enumerate(events):
                                    valid = [f for f in ev if 1 <= f <= num_frames]
                                    if not valid:
                                        continue
                                    avg = sum(per_frame[f - 1].item() for f in valid) / len(valid)
                                    if avg > best_avg:
                                        best_avg = avg
                                        best_idx = ei
                            if use_dominant and layer_idx == first_vf_layer and best_idx >= 0:
                                self._dominant_event_idx = best_idx
                        if best_idx is not None and best_idx >= 0:
                            boost = self._vision_focus_boost
                            for f in events[best_idx]:
                                if f < 1 or f > num_frames:
                                    continue
                                fs = vid_start + (f - 1) * tpf
                                fe = vid_start + f * tpf
                                fs = max(0, min(fs, kv_len))
                                fe = max(fs, min(fe, kv_len))
                                if fs < fe:
                                    attn_weights[:, :, :, fs:fe] = attn_weights[:, :, :, fs:fe] + boost

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=module.attention_dropout, training=module.training)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, module.hidden_size)
            attn_output = module.o_proj(attn_output)
            if not output_attentions:
                attn_weights = None
            return attn_output, attn_weights, past_key_value

        return forward

    def install_patches(self, layers: range | None = None):
        self.remove_patches()
        num_layers = len(self.model.model.layers)
        if layers is None:
            layers = range(num_layers)
        for i in layers:
            attn_module = self.model.model.layers[i].self_attn
            original_forward = attn_module.forward
            self._patched_forwards[i] = original_forward
            attn_module.forward = self._make_patched_forward(original_forward, i)
        logger.info(f"Forward patch on layers {list(layers)}.")

    def remove_patches(self):
        for i, orig_fwd in self._patched_forwards.items():
            self.model.model.layers[i].self_attn.forward = orig_fwd
        if self._patched_forwards:
            logger.info(f"Restored forward on layers {list(self._patched_forwards.keys())}.")
        self._patched_forwards.clear()
        self._token_info = None
        self._events = []
        self._event_bias_value = 0.0
        self._vision_focus_boost = 0.0
        self._features = set()
        self._dominant_event_idx = None

    def _get_frame_embeddings(self, video_tensor: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            encoded = self.model.encode_images(video_tensor)
            pooled = self.model.get_2dPool(encoded)
            frame_embeds = pooled.mean(dim=1)
        return frame_embeds

    def _compute_event_boundaries(
        self, frame_embeds: torch.Tensor, threshold: float = 0.5
    ) -> list[list[int]]:
        if frame_embeds.shape[0] <= 1:
            return [[i for i in range(1, frame_embeds.shape[0] + 1)]]
        a = frame_embeds[:-1]
        b = frame_embeds[1:]
        sim = torch.nn.functional.cosine_similarity(a, b, dim=1)
        boundaries = [0]
        for i in range(len(sim)):
            if sim[i].item() < threshold:
                boundaries.append(i + 1)
        boundaries.append(frame_embeds.shape[0])
        events = []
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            events.append([i for i in range(s + 1, e + 1)])
        return events

    def run_inference(
        self,
        video_id: str,
        video_path: str,
        question: str = "Describe the video in detail.",
        event_similarity_threshold: float = 0.5,
        event_bias_value: float = -5.0,
        cross_event_layers: list | None = None,
        vision_focus_layers: list | None = None,
        features: list | None = None,
        vision_focus_boost: float = 0.5,
        output_dir: str | None = None,
        save_result: bool = True,
    ) -> dict | None:
        """Basic LLaVA-NeXT video inference: load video, prompt, generate, decode."""
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

        # get event boundaries
        
        with torch.inference_mode():
            frame_embeds = self._get_frame_embeddings(video[0])
            events = self._compute_event_boundaries(frame_embeds, event_similarity_threshold)
            logger.info(f"[Events] {events}")

        video_index = torch.where(input_ids == IMAGE_TOKEN_INDEX)[-1][0].item()
        token_info = {
            "video_index": video_index,
            "original_ids_len": input_ids.shape[1],
            "num_frames": self.for_get_frames_num,
            "token_per_frame": None,
            "visual_token_len": None,
        }

        num_layers = len(self.model.model.layers)

        def _range_from_cfg(cfg: list | None) -> range:
            if cfg is None or len(cfg) < 2:
                return range(num_layers)
            start, end = cfg[0], cfg[1]
            return range(max(0, start), min(num_layers, end + 1))

        feature_set = set(features or [])

        ce_range = range(0)
        vf_range = range(0)
        if "cross_event_boundary" in feature_set or "cross_event" in feature_set:
            ce_range = _range_from_cfg(cross_event_layers)
        if "vision_focus_greedy" in feature_set or "vision_focus_dominant" in feature_set:
            vf_range = _range_from_cfg(vision_focus_layers)

        layers_set = set(ce_range) | set(vf_range)
        if layers_set:
            layers_range = sorted(layers_set)
            self.install_patches(layers=layers_range)
        else:
            layers_range = []

        self._cross_event_layers = set(ce_range)
        self._vision_focus_layers = set(vf_range)
        self._token_info = token_info
        self._events = events
        self._event_bias_value = event_bias_value
        self._vision_focus_boost = vision_focus_boost
        self._features = set(features or [])

        with torch.inference_mode():
            _name = (
                getattr(self.cfg_pretrained, "_name_or_path", "")
                or self.model_path
            )
            is_mistral = "mistral" in _name.lower()
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
            }
            if not is_mistral:
                generate_kwargs["stopping_criteria"] = [stopping_criteria]

            output = self.model.generate(**generate_kwargs)

        self.remove_patches()

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
    parser.add_argument("--video_path", type=str, default="/path/to/video.mp4")
    parser.add_argument("--video_id", type=str, default="result")
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION)
    parser.add_argument("--features", nargs="+", choices=["cross_event", "cross_event_boundary", "vision_focus_greedy", "vision_focus_dominant"], default=["cross_event_boundary"])

    args = parser.parse_args()

    manipulator = NoahLLaVANeXTVideo(
        model_path="lmms-lab/LLaVA-NeXT-Video-7B",
        model_base=None,
        conv_mode="vicuna_v1",
        mm_spatial_pool_stride=4,
        mm_spatial_pool_mode="average",
        overwrite=True,
        for_get_frames_num=8,
    )

    manipulator.run_inference(
        video_id=args.video_id,
        video_path=args.video_path,
        question=args.question,
        event_similarity_threshold=DEFAULT_CONFIG["event_similarity_threshold"],
        event_bias_value=DEFAULT_CONFIG["cross_event_bias_value"],
        cross_event_layers=DEFAULT_CONFIG["cross_event_layers"],
        vision_focus_layers=DEFAULT_CONFIG["vision_focus_layers"],
        features=args.features,
        vision_focus_boost=DEFAULT_CONFIG["vision_focus_boost"],
    )


