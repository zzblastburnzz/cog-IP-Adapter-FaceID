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

base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
base_cache = "model-cache"
vae_model_path = "stabilityai/sd-vae-ft-mse"
ip_cache = "./ip-cache"
device = "cuda"

class Predictor(BasePredictor):
    def setup(self) -> None:
        if not os.path.exists("ip-cache/ip-adapter-faceid_sd15.bin"):
            os.makedirs(ip_cache)
            os.system(
                "wget -O ip-cache/ip-adapter-faceid_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin"
            )

        self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

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

        self.ip_model = IPAdapterFaceID(
            pipe,
            "ip-cache/ip-adapter-faceid_sd15.bin",
            device
        )

    @torch.inference_mode()
    def predict(
        self,
        face_image: Path = Input(description="Input face image"),
        gender: str = Input(description="Giới tính nhân vật (male hoặc female)", default="female"),
        job: str = Input(description="Nghề nghiệp (ví dụ: kiếm tiên, đạo sĩ, đan sư...)", default="kiếm tiên"),
        vip_level: int = Input(description="Cấp độ VIP (ảnh hưởng ánh sáng và trang phục)", default=0),
        prompt: Optional[str] = Input(
            description="Override prompt. Nếu để trống sẽ sinh tự động theo giới tính + nghề + VIP",
            default=None
        ),
        negative_prompt: str = Input(
            description="Negative prompt để tránh lỗi hình ảnh",
            default="monochrome, lowres, blurry, bad anatomy, extra hands, watermark, cropped, low quality, duplicated limbs",
        ),
        width: int = Input(description="Chiều rộng ảnh", default=768),
        height: int = Input(description="Chiều cao ảnh", default=1366),
        num_outputs: int = Input(description="Số ảnh tạo", ge=1, le=4, default=1),
        num_inference_steps: int = Input(description="Số bước dựng ảnh", ge=1, le=200, default=30),
        seed: Optional[int] = Input(description="Random seed", default=None),
        agree_to_research_only: bool = Input(
            description="You must agree to use this model only for research. It is not for commercial use.",
            default=False,
        ),
    ) -> List[Path]:
        if not agree_to_research_only:
            raise Exception("Bạn phải đồng ý chỉ dùng model này cho mục đích nghiên cứu, không thương mại.")

        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")

        self.app.prepare(ctx_id=0, det_size=(640, 640))
        image = cv2.imread(str(face_image))
        faces = self.app.get(image)

        if not faces:
            raise Exception("Không phát hiện khuôn mặt trong ảnh.")

        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

        # 👉 Tạo prompt tự động nếu không nhập
        if not prompt or prompt.strip() == "":
            gender_tag = "beautiful female cultivator" if gender.lower() == "female" else "handsome male cultivator"
            vip_tag = f"luxury shining outfit, glowing energy, vip level {vip_level}" if vip_level > 0 else ""
            prompt = f"{gender_tag}, {job}, chinese fantasy style, full body, cinematic lighting, intricate costume, {vip_tag}"

        images = self.ip_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            faceid_embeds=faceid_embeds,
            num_samples=num_outputs,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            seed=seed
        )

        output_paths = []
        for i, image in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
