# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import cv2
import torch
from PIL import Image
from typing import List, Optional

from insightface.app import FaceAnalysis
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL

# ----- Config -----
base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
base_cache = "model-cache"
ip_cache = "ip-cache"
device = "cuda"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load all heavy models once."""
        # Ensure cache folders exist
        os.makedirs(base_cache, exist_ok=True)
        os.makedirs(ip_cache, exist_ok=True)

        # Download IP-Adapter FaceID weight if missing (use curl for portability)
        ip_faceid_path = os.path.join(ip_cache, "ip-adapter-faceid_sd15.bin")
        if not os.path.exists(ip_faceid_path):
            exit_code = os.system(
                'curl -L -o "ip-cache/ip-adapter-faceid_sd15.bin" '
                'https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin'
            )
            if exit_code != 0 or not os.path.exists(ip_faceid_path):
                raise RuntimeError("Failed to download IP-Adapter FaceID checkpoint.")

        # Init InsightFace (prefer GPU, fallback CPU gracefully)
        try:
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception:
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )

        # Scheduler (SD1.5 style)
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        # Load VAE + SD pipeline
        vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None,
            cache_dir=base_cache,
        )
        self.pipe = pipe.to(device)

        # Bind IP-Adapter FaceID to the SD pipeline
        self.ip_model = IPAdapterFaceID(
            self.pipe,
            ip_faceid_path,
            device,
        )

    @torch.inference_mode()
    def predict(
        self,
        face_image: Path = Input(description="Ảnh khuôn mặt đầu vào (jpg/png)"),
        gender: str = Input(description="Giới tính nhân vật (male|female)", default="female"),
        job: str = Input(description="Nghề nghiệp (kiếm tiên, đạo sĩ, đan sư...)", default="kiếm tiên"),
        vip_level: int = Input(description="Cấp độ VIP (tăng ánh sáng/trang phục)", default=0),
        prompt: Optional[str] = Input(
            description="Prompt tùy chỉnh. Để trống sẽ tự sinh theo giới tính + nghề + VIP",
            default=None,
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="monochrome, lowres, blurry, bad anatomy, extra hands, watermark, cropped, low quality, duplicated limbs",
        ),
        width: int = Input(description="Chiều rộng", default=768),
        height: int = Input(description="Chiều cao", default=1366),
        num_outputs: int = Input(description="Số ảnh tạo", ge=1, le=4, default=1),
        num_inference_steps: int = Input(description="Số bước dựng", ge=1, le=200, default=30),
        seed: Optional[int] = Input(description="Random seed", default=None),
        agree_to_research_only: bool = Input(
            description="Đồng ý chỉ dùng cho nghiên cứu (không thương mại).",
            default=False,
        ),
    ) -> List[Path]:
        if not agree_to_research_only:
            raise Exception("Bạn phải đồng ý chỉ dùng model này cho mục đích nghiên cứu, không thương mại.")

        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"[predict] Using seed: {seed}")

        # Prepare face detector (ctx_id=0 for GPU if available)
        try:
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception:
            # Fallback CPU context if GPU provider unavailable for InsightFace
            self.app.prepare(ctx_id=-1, det_size=(640, 640))

        # Read and detect face
        image = cv2.imread(str(face_image))
        if image is None:
            raise Exception("Không đọc được ảnh đầu vào.")
        faces = self.app.get(image)
        if not faces:
            raise Exception("Không phát hiện khuôn mặt trong ảnh.")

        # Use the first detected face
        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

        # Auto prompt if empty
        if not prompt or prompt.strip() == "":
            gender_tag = (
                "beautiful female cultivator"
                if gender.lower() == "female"
                else "handsome male cultivator"
            )
            vip_tag = (
                f"luxury shining outfit, glowing energy, vip level {vip_level}"
                if vip_level > 0
                else ""
            )
            prompt = f"{gender_tag}, {job}, chinese fantasy style, full body, cinematic lighting, intricate costume, {vip_tag}"

        # Generate
        images = self.ip_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            faceid_embeds=faceid_embeds,
            num_samples=num_outputs,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

        # Save outputs
        output_paths: List[Path] = []
        for i, img in enumerate(images):
            out = f"/tmp/out-{i}.png"
            img.save(out)
            output_paths.append(Path(out))

        return output_paths
