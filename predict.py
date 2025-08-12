from cog import BasePredictor, Input, Path
import os
import cv2
import torch
from typing import List
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from diffusers import StableDiffusionPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler

# ==== Set biến môi trường ngay trong code ====
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BASE_MODEL = "SG161222/Realistic_Vision_V4.0_noVAE"
VAE_MODEL = "stabilityai/sd-vae-ft-mse"
IP_CACHE = "./ip-cache"
IP_REPO = "h94/IP-Adapter-FaceID"
IP_FILE = "ip-adapter-faceid_sd15.bin"

def _ensure_multiple_of_8(x: int) -> int:
    return max(64, (x // 8) * 8)

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Tải checkpoint bằng HuggingFace Hub
        os.makedirs(IP_CACHE, exist_ok=True)
        ckpt_path = os.path.join(IP_CACHE, IP_FILE)
        if not os.path.exists(ckpt_path):
            ckpt_path = hf_hub_download(
                repo_id=IP_REPO,
                filename=IP_FILE,
                local_dir=IP_CACHE,
                local_dir_use_symlinks=False,
            )

        # InsightFace
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # Stable Diffusion pipeline
        vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=self.torch_dtype)
        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=self.torch_dtype,
            vae=vae,
            safety_checker=None,
            feature_extractor=None,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe.to(self.device)

        # IP-Adapter FaceID
        self.ip_model = IPAdapterFaceID(self.pipe, ckpt_path, self.device)

    @torch.inference_mode()
    def predict(
        self,
        face_image: Path = Input(description="Ảnh mặt (jpg/png)"),
        prompt: str = Input(
            default=(
                "chân dung phong cách tiên hiệp, cổ trang, ánh sáng mềm, hậu cảnh mờ, "
                "cinematic, highly detailed, ultra-detailed skin, 8k, masterpiece"
            )
        ),
        negative_prompt: str = Input(
            default="monochrome, lowres, bad anatomy, worst quality, low quality, blurry, multiple people, deformed, oversmooth skin"
        ),
        vip_level: int = Input(default=0, ge=0, le=10, description="Cấp VIP (0 = không VIP)"),
        job: str = Input(default="", description="Nghề nghiệp nhân vật, ví dụ: kiếm khách, pháp sư, y sư..."),
        width: int = Input(default=720, ge=64, le=2048, description="Mặc định 9:16 (720x1280)"),
        height: int = Input(default=1280, ge=64, le=2048),
        num_outputs: int = Input(default=1, ge=1, le=4),
        num_inference_steps: int = Input(default=30, ge=5, le=200),
        guidance_scale: float = Input(default=5.0, ge=0.0, le=20.0),
        seed: int = Input(default=None),
    ) -> List[Path]:
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        print(f"[IPAdapter] Using seed: {seed}")

        # Ghép VIP và nghề nghiệp vào prompt
        extra_tags = []
        if vip_level > 0:
            extra_tags.append(f"VIP cấp {vip_level}")
        if job.strip():
            extra_tags.append(f"nghề {job.strip()}")
        if extra_tags:
            prompt = f"{prompt}, " + ", ".join(extra_tags)

        # Đọc ảnh & detect face
        image = cv2.imread(str(face_image))
        if image is None:
            raise ValueError("Không đọc được ảnh đầu vào.")
        faces = self.app.get(image)
        if not faces:
            raise ValueError("Không phát hiện khuôn mặt nào trong ảnh.")
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)
        face = faces[0]

        # Lấy embedding FaceID
        faceid_embeds = torch.tensor(
            face.normed_embedding, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Ép về tỉ lệ 9:16
        if width * 16 != height * 9:
            height = max(height, 1280)
            width = (height * 9) // 16
        width = _ensure_multiple_of_8(width)
        height = _ensure_multiple_of_8(height)

        # Sinh ảnh
        images = self.ip_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            faceid_embeds=faceid_embeds,
            num_samples=num_outputs,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            generator=generator,
        )

        # Lưu output
        output_paths: List[Path] = []
        for i, img in enumerate(images):
            out_path = f"/tmp/out-{i}.png"
            img.save(out_path)
            output_paths.append(Path(out_path))

        return output_paths
