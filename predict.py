# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
from cog import BasePredictor, Input, Path
from PIL import Image
import math
import os
from glob import glob
from pathlib import Path as PathlibPath
import uuid

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf
from torchvision.transforms import ToTensor

# local imports
from utils.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from utils.helpers import embed_watermark, default, instantiate_from_config
from utils.sizing_strategy import SizingStrategy


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.sizing_strategy = SizingStrategy()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.svd_num_frames = 14
        svd_num_steps = 25
        svd_model_config = "./configs/svd.yaml"

        svd_model, svd_filter = self.load_model(
            svd_model_config,
            self.device,
            self.svd_num_frames,
            svd_num_steps,
        )
        self.svd_model = svd_model
        self.svd_filter = svd_filter

        self.svdxt_num_frames = 25
        svdxt_num_steps = 30
        svdxt_model_config = "./configs/svd_xt.yaml"
        svdxt_model, svdxt_filter = self.load_model(
            svdxt_model_config,
            self.device,
            self.svdxt_num_frames,
            svdxt_num_steps,
        )
        self.svdxt_model = svdxt_model
        self.svdxt_filter = svdxt_filter

    @torch.inference_mode()
    def predict(
        self,
        input_image: Path = Input(
            description="Path to the input image file or folder with image files",
        ),
        num_frames: int = Input(
            description="Number of frames to process",
            default=None
        ),
        version: str = Input(
            description="Version of the model",
            choices=[
                "svd",
                "svd_xt",
            ],
            default="svd"
        ),
        sizing_strategy: str = Input(
            description="Decide how to resize the input image",
            choices=[
                "maintain_aspect_ratio",
                "crop_to_16_9",
                "use_image_dimensions",
            ],
            default="maintain_aspect_ratio",
        ),
        motion_bucket_id: int = Input(
            description="Motion bucket ID for video processing",
            default=127
        ),
        cond_aug: float = Input(
            description="Condition augmentation factor",
            default=0.02,
            ge=0
        ),
        seed: int = Input(
            description="Seed for random number generation",
            default=23
        ),
        fps_id: int = Input(description="Frames per second",
                            default=6, ge=5, le=30),
        decoding_t: int = Input(
            description="Number of frames decoded at a time, affects VRAM usage",
            default=14,
            ge=0
        ),
    ) -> Path:
        # check that the version is valid
        output_folder: Optional[str] = "tmp/"
        if version != "svd" and version != "svd_xt":
            raise ValueError(f"Version {version} does not exist.")
        torch.manual_seed(seed)

        # set the model and filter
        model = self.svd_model if version == "svd" else self.svdxt_model
        filter = self.svd_filter if version == "svd" else self.svdxt_filter

        if version == "svd":
            num_frames = default(num_frames, self.svd_num_frames)
        elif version == "svd_xt":
            num_frames = default(num_frames, self.svdxt_num_frames)

        image = self.sizing_strategy.apply(sizing_strategy, input_image)

        if image.mode == "RGBA":
            image = image.convert("RGB")
        w, h = image.size

        if h % 64 != 0 or w % 64 != 0:
            width, height = map(lambda x: x - x % 64, (w, h))
            image = image.resize((width, height))
            print(
                f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
            )

        image = ToTensor()(image)
        image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(self.device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024):
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )

        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")

        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        value_dict["cond_aug"] = cond_aug

        output_path = None
        with torch.no_grad():
            with torch.autocast(self.device):
                batch, batch_uc = self.get_batch(
                    self.get_unique_embedder_keys_from_conditioner(
                        model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=self.device,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(
                        uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(
                        c[k], "b t ... -> (b t) ...", t=num_frames)

                randn = torch.randn(shape, device=self.device)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(self.device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp(
                    (samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                os.makedirs(output_folder, exist_ok=True)
                base_count = len(glob(os.path.join(output_folder, "*.mp4")))
                video_path = os.path.join(
                    output_folder, f"{base_count:06d}.mp4")

                output_path = video_path

                samples = embed_watermark(samples)
                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                # Save frames as individual images
                for i, frame in enumerate(vid):
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(
                        os.path.join(
                            output_folder, f"frame_{i:06d}.png"), frame
                    )

                # Use ffmpeg to create video from images
                os.system(
                    f"ffmpeg -r {fps_id + 1} -i {output_folder}/frame_%06d.png -c:v libx264 -vf 'fps={fps_id + 1},format=yuv420p' {video_path}"
                )

                # Remove individual frame images
                for file_name in glob(os.path.join(output_folder, "*.png")):
                    os.remove(file_name)
        return Path(output_path)

    def get_unique_embedder_keys_from_conditioner(self, conditioner):
        return list(set([x.input_key for x in conditioner.embedders]))

    def get_batch(self, keys, value_dict, N, T, device):
        batch = {}
        batch_uc = {}

        for key in keys:
            if key == "fps_id":
                batch[key] = (
                    torch.tensor([value_dict["fps_id"]])
                    .to(device)
                    .repeat(int(math.prod(N)))
                )
            elif key == "motion_bucket_id":
                batch[key] = (
                    torch.tensor([value_dict["motion_bucket_id"]])
                    .to(device)
                    .repeat(int(math.prod(N)))
                )
            elif key == "cond_aug":
                batch[key] = repeat(
                    torch.tensor([value_dict["cond_aug"]]).to(device),
                    "1 -> b",
                    b=math.prod(N),
                )
            elif key == "cond_frames":
                batch[key] = repeat(value_dict["cond_frames"],
                                    "1 ... -> b ...", b=N[0])
            elif key == "cond_frames_without_noise":
                batch[key] = repeat(
                    value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
                )
            else:
                batch[key] = value_dict[key]

        if T is not None:
            batch["num_video_frames"] = T

        for key in batch.keys():
            if key not in batch_uc and isinstance(batch[key], torch.Tensor):
                batch_uc[key] = torch.clone(batch[key])
        return batch, batch_uc

    def load_model(
        self,
        config: str,
        device: str,
        num_frames: int,
        num_steps: int,
    ):
        config = OmegaConf.load(config)
        if device == "cuda":
            config.model.params.conditioner_config.params.emb_models[
                0
            ].params.open_clip_embedding_config.params.init_device = device

        config.model.params.sampler_config.params.num_steps = num_steps
        config.model.params.sampler_config.params.guider_config.params.num_frames = (
            num_frames
        )
        if device == "cuda":
            with torch.device(device):
                model = instantiate_from_config(config.model).to(device).eval()
        else:
            model = instantiate_from_config(config.model).to(device).eval()

        filter = DeepFloydDataFiltering(verbose=False, device=device)
        return model, filter
