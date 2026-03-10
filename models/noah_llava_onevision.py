import argparse
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

DEFAULT_CONFIG = {
    "event_similarity_threshold": 0.94,
    "cross_event_bias_value": -3.0,
    "cross_event_layers": [0, 15],
    "vision_focus_boost": 5.0,
    "vision_focus_layers": [8, 23],
}

DEFAULT_QUESTION = "What happens in this video? Please describe it in detail."

class NoahLLaVAOneVision:
    def __init__(
        self,
        model_path,
        model_base=None,
        vision_tower_path=None,
        conv_mode="qwen_1_5",
        for_get_frames_num=16,
        device_map="auto",
        output_dir="/home/work/Redteaming/kyuho/noah/output",
    ):
        self.model_path = model_path
        self.model_base = model_base
        self.vision_tower_path = vision_tower_path
        self.conv_mode = conv_mode
        self.for_get_frames_num = for_get_frames_num
        self.device_map = device_map
        self.output_dir = output_dir
        self._patched_forwards = {}
        self._token_info = None
        self._events: list[list[int]] = []
        self._event_bias_value: float = 0.0
        self._cross_event_layers: set[int] = set()
        self._features: set[str] = set()

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
            torch_dtype="bfloat16",
            attn_implementation="eager",
            **llava_model_args,
        )
        self.model.eval()

    def _make_patched_forward(self, original_forward, layer_idx):
        """Patched forward based on Qwen2Attention.forward (eager) with cross-event bias."""
        module = self.model.model.layers[layer_idx].self_attn

        def forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()

            query_states = module.q_proj(hidden_states)
            key_states = module.k_proj(hidden_states)
            value_states = module.v_proj(hidden_states)

            query_states = query_states.view(
                bsz, q_len, module.num_heads, module.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, module.num_key_value_heads, module.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, module.num_key_value_heads, module.head_dim
            ).transpose(1, 2)

            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                cos, sin = module.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            if past_key_value is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_value.update(
                    key_states, value_states, module.layer_idx, cache_kwargs
                )

            key_states = repeat_kv(key_states, module.num_key_value_groups)
            value_states = repeat_kv(value_states, module.num_key_value_groups)

            # Same as Qwen2Attention.forward: only add mask when attention_mask is provided
            attn_weights = (
                torch.matmul(query_states, key_states.transpose(2, 3))
                / math.sqrt(module.head_dim)
            )
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # ------------------------------------------------------------------
            # Cross-event attention bias (prefill only, q_len > 1)
            # ------------------------------------------------------------------
            if (
                "cross_event" in self._features
                or "cross_event_boundary" in self._features
            ) and q_len > 1:
                if hasattr(self, "_cross_event_layers") and layer_idx in self._cross_event_layers:
                    info = self._token_info
                    events = self._events
                    bias_value = self._event_bias_value
                    kv_len = key_states.shape[-2]

                    if info and events and len(events) > 1 and bias_value != 0:
                        # Lazily infer token_per_frame / visual_token_len from current sequence
                        if info.get("token_per_frame") is None:
                            vis_len = q_len - (info["original_ids_len"] - 1)
                            if info["num_frames"] > 0:
                                info["token_per_frame"] = vis_len // info["num_frames"]
                                info["visual_token_len"] = vis_len

                        vid_start = info["video_index"]
                        tpf = info.get("token_per_frame")
                        num_frames = info.get("num_frames")

                        if tpf and num_frames and vid_start + num_frames * tpf <= kv_len:
                            bias = torch.zeros(
                                1,
                                1,
                                q_len,
                                kv_len,
                                device=attn_weights.device,
                                dtype=attn_weights.dtype,
                            )

                            # cross_event_boundary: only consecutive events
                            if "cross_event_boundary" in self._features:
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
                                    src_s = max(0, src_s)
                                    src_e = min(src_e, q_len, kv_len)
                                    tgt_s = max(0, tgt_s)
                                    tgt_e = min(tgt_e, q_len, kv_len)
                                    if src_s < src_e and tgt_s < tgt_e:
                                        bias[0, 0, src_s:src_e, tgt_s:tgt_e] += bias_value

                            # cross_event: later events attending to earlier events
                            if "cross_event" in self._features:
                                for i in range(len(events)):
                                    for j in range(i + 1, len(events)):
                                        earlier_event, later_event = events[i], events[j]
                                        for src_frame in later_event:
                                            for tgt_frame in earlier_event:
                                                if not (
                                                    1 <= src_frame <= num_frames
                                                    and 1 <= tgt_frame <= num_frames
                                                ):
                                                    continue
                                                src_s = vid_start + (src_frame - 1) * tpf
                                                src_e = vid_start + src_frame * tpf
                                                tgt_s = vid_start + (tgt_frame - 1) * tpf
                                                tgt_e = vid_start + tgt_frame * tpf
                                                src_s = max(0, src_s)
                                                src_e = min(src_e, q_len, kv_len)
                                                tgt_s = max(0, tgt_s)
                                                tgt_e = min(tgt_e, q_len, kv_len)
                                                if src_s < src_e and tgt_s < tgt_e:
                                                    bias[0, 0, src_s:src_e, tgt_s:tgt_e] += bias_value

                            attn_weights = attn_weights + bias

            # Upcast attention to fp32 (same as Qwen2Attention)
            attn_weights = F.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights = F.dropout(
                attn_weights,
                p=module.attention_dropout,
                training=module.training,
            )
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, module.num_heads, q_len, module.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, module.num_heads, q_len, module.head_dim)}, but is {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, module.hidden_size)

            attn_output = module.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None
            return attn_output, attn_weights, past_key_value

        return forward

    def install_patches(self, layers=None):
        self.remove_patches()
        num_layers = len(self.model.model.layers)
        if layers is None:
            layers = range(num_layers)
        for i in layers:
            attn_module = self.model.model.layers[i].self_attn
            original_forward = attn_module.forward
            self._patched_forwards[i] = original_forward
            attn_module.forward = self._make_patched_forward(original_forward, i)

    def remove_patches(self):
        for i, orig_fwd in self._patched_forwards.items():
            self.model.model.layers[i].self_attn.forward = orig_fwd
        self._patched_forwards.clear()
        self._token_info = None
        self._events = []
        self._event_bias_value = 0.0
        self._cross_event_layers = set()
        self._features = set()

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

    def _get_frame_embeddings(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames with the same pipeline as LLaVA (vision_tower + projector + get_2dPool),
        then return one embedding per frame: (num_frames, hidden_size).
        """
        with torch.inference_mode():
            encoded = self.model.encode_images(video_tensor)
            pooled = self.model.get_2dPool(encoded)
            frame_embeds = pooled.mean(dim=1)
        return frame_embeds

    def _compute_event_boundaries(
        self, frame_embeds: torch.Tensor, threshold: float = 0.5
    ) -> list[list[int]]:
        """
        Consecutive cosine similarity; where sim[i] < threshold, put a boundary between frame i and i+1.
        Returns list of events, each event = list of 1-indexed frame numbers.
        """
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
        events: list[list[int]] = []
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            events.append([i for i in range(s + 1, e + 1)])
        return events

    def _range_from_cfg(self, cfg: list | None) -> range:
        """Helper: convert [start, end] style config into a layer range."""
        num_layers = len(self.model.model.layers)
        if cfg is None or len(cfg) < 2:
            return range(num_layers)
        start, end = cfg[0], cfg[1]
        return range(max(0, start), min(num_layers, end + 1))

    def run_inference(
        self,
        video_id,
        video_path,
        question="Describe what's happening in this video.",
        max_new_tokens=4096,
        output_dir=None,
        event_similarity_threshold=DEFAULT_CONFIG["event_similarity_threshold"],
        event_bias_value: float = DEFAULT_CONFIG["cross_event_bias_value"],
        cross_event_layers: list | None = None,
        features: list | None = None,
        save_result=False,
    ):
        if output_dir is None:
            output_dir = self.output_dir

        feature_set = set(features or [])

        video_frames = self.load_video(video_path)
        image_tensors = []
        frames = (
            self.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
            .to(device="cuda", dtype=self.model.dtype)
        )
        image_tensors.append(frames)

        with torch.inference_mode():
            frame_embeds = self._get_frame_embeddings(frames)
            events = self._compute_event_boundaries(
                frame_embeds,
                threshold=event_similarity_threshold,
            )
        logger.info(f"[Events] {events}")

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

        token_info = {
            "video_index": torch.where(input_ids == IMAGE_TOKEN_INDEX)[-1][0].item(),
            "original_ids_len": input_ids.shape[1],
            "num_frames": self.for_get_frames_num,
            "token_per_frame": None,
            "visual_token_len": None,
        }

        ce_range = range(0)
        if "cross_event" in feature_set or "cross_event_boundary" in feature_set:
            cfg = cross_event_layers or DEFAULT_CONFIG["cross_event_layers"]
            ce_range = self._range_from_cfg(cfg)

        layers_range = sorted(set(ce_range))
        if layers_range:
            self.install_patches(layers=layers_range)

        # Save state for patched forwards
        self._cross_event_layers = set(ce_range)
        self._token_info = token_info
        self._events = events
        self._event_bias_value = event_bias_value
        self._features = feature_set

        try:
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
        finally:
            if layers_range:
                self.remove_patches()

        generated_text = self.tokenizer.batch_decode(
            output, skip_special_tokens=True
        )[0].strip()

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
    parser.add_argument(
        "--video_path",
        type=str,
        default="/home/work/Redteaming/kyuho/noah/dataset/noah/zcDA0s8eWU4_m978SIFnHS8_1_mid_center.mp4",
    )
    parser.add_argument("--video_id", type=str, default="demo")
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION)
    parser.add_argument(
        "--features",
        nargs="+",
        choices=["cross_event", "cross_event_boundary"],
        default=["cross_event_boundary"],
    )

    args = parser.parse_args()

    manipulator = NoahLLaVAOneVision(
        model_path="/home/work/Redteaming/data1/noah/llava-onevision-qwen2-7b-ov",
        model_base=None,
        vision_tower_path="/home/work/Redteaming/data1/noah/siglip-so400m-patch14-384",
        for_get_frames_num=8,
    )

    result = manipulator.run_inference(
        video_id=args.video_id,
        video_path=args.video_path,
        question=args.question,
        event_similarity_threshold=DEFAULT_CONFIG["event_similarity_threshold"],
        event_bias_value=DEFAULT_CONFIG["cross_event_bias_value"],
        cross_event_layers=DEFAULT_CONFIG["cross_event_layers"],
        features=args.features,
    )
    print(result["answer"])
