from cog import BasePredictor, Input, Path
import os
import cv2
import torch
from typing import List
from insightface.app import FaceAnalysis
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from diffusers import StableDiffusionPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler

BASE_MODEL = "SG161222/Realistic_Vision_V4.0_noVAE"
VAE_MODEL = "stabilityai/sd-vae-ft-mse"
IP_CACHE = "./ip-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        # hạn chế thread BLAS cho môi trường nhỏ
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        os.makedirs(IP_CACHE, exist_ok=True)
        ckpt_path = os.path.join(IP_CACHE, "ip-adapter-faceid_sd15.bin")
        if not os.path.exists(ckpt_path):
            os.system(
                f"wget -q -O {ckpt_path} "
                "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin"
            )

        # dùng CPUExecutionProvider để tránh lỗi CUDA lib trong môi trường không có GPU
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

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

        self.ip_model = IPAdapterFaceID(self.pipe, ckpt_path, self.device)

    @torch.inference_mode()
    def predict(
        self,
        face_image: Path = Input(description="Ảnh khuôn mặt đầu vào (jpg/png)"),
        prompt: str = Input(
            default=(
                "1girl, ancient fantasy portrait, Chinese xianxia style, hanfu, flowing fabrics, "
                "ornate hairpins, soft ethereal lighting, detailed skin, high realism, cinematic grade"
            ),
            description="Prompt chính (mặc định phong cách tiên hiệp/cổ trang)"
        ),
        negative_prompt: str = Input(
            default=(
                "lowres, bad anatomy, worst quality, low quality, blurry, deformed, extra limbs, "
                "text, watermark, logo, multiple people"
            )
        ),
        # 9:16, bội số của 16 để hợp SD1.5: 832x1472 ≈ 0.565 (gần 0.5625)
        width: int = Input(default=832, ge=64, le=2048, description="Chiều ngang (mặc định 9:16)"),
        height: int = Input(default=1472, ge=64, le=2048, description="Chiều dọc (mặc định 9:16)"),
        num_outputs: int = Input(default=1, ge=1, le=4),
        num_inference_steps: int = Input(default=30, ge=1, le=200),
        seed: int = Input(default=None),
        agree_to_research_only: bool = Input(default=True),
    ) -> List[Path]:
        if not agree_to_research_only:
            raise Exception("Bạn phải đồng ý chỉ dùng cho mục đích nghiên cứu.")

        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        print(f"Using seed: {seed}")

        # đảm bảo đúng tỉ lệ 9:16 (nếu người dùng sửa width/height)
        def snap_to_9_16(w: int, h: int) -> tuple[int, int]:
            # ép về bội số của 16 và giữ gần 9:16 nhất
            w = max(64, (w // 16) * 16)
            h = max(64, (h // 16) * 16)
            target = 9 / 16
            r = w / h
            if abs(r - target) > 0.02:
                # scale theo chiều cao để đạt gần 9:16
                w = int(round(h * target / 16)) * 16
            return w, h

        width, height = snap_to_9_16(width, height)

        image = cv2.imread(str(face_image))
        if image is None:
            raise ValueError("Không đọc được ảnh đầu vào.")
        faces = self.app.get(image)
        if not faces:
            raise ValueError("Không phát hiện được khuôn mặt nào trong ảnh.")

        faceid_embeds = torch.tensor(
            faces[0].normed_embedding, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        images = self.ip_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            faceid_embeds=faceid_embeds,
            num_samples=num_outputs,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            seed=seed,
            generator=generator,
        )

        outputs: List[Path] = []
        for i, img in enumerate(images):
            out_path = f"/tmp/out-{i}.png"
            img.save(out_path)
            outputs.append(Path(out_path))
        return outputs
