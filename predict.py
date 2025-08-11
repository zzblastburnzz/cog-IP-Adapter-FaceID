from cog import BasePredictor, Input, Path
import os
import cv2
import torch
from PIL import Image
from typing import List
from insightface.app import FaceAnalysis
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from diffusers import StableDiffusionPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler

base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
base_cache = "model-cache"
vae_model_path = "stabilityai/sd-vae-ft-mse"
ip_cache = "./ip-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        os.makedirs(ip_cache, exist_ok=True)
        faceid_ckpt = os.path.join(ip_cache, "ip-adapter-faceid_sd15.bin")
        if not os.path.exists(faceid_ckpt):
            os.system(
                f"wget -O {faceid_ckpt} https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin"
            )

        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        vae = AutoencoderKL.from_pretrained(vae_model_path, torch_dtype=self.torch_dtype)
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=self.torch_dtype,
            vae=vae,
            safety_checker=None,
            feature_extractor=None,
            cache_dir=base_cache,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

        self.pipe = pipe.to(self.device)
        self.ip_model = IPAdapterFaceID(self.pipe, faceid_ckpt, self.device)

    @torch.inference_mode()
    def predict(
        self,
        face_image: Path = Input(description="Input face image"),
        prompt: str = Input(description="Prompt", default="photo of a woman in red dress in a garden"),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="monochrome, lowres, bad anatomy, worst quality, low quality, blurry, multiple people"
        ),
        width: int = Input(default=1024, ge=64, le=2048),
        height: int = Input(default=1024, ge=64, le=2048),
        num_outputs: int = Input(default=1, ge=1, le=4),
        num_inference_steps: int = Input(default=30, ge=1, le=200),
        seed: int = Input(default=None),
        agree_to_research_only: bool = Input(default=False)
    ) -> List[Path]:
        if not agree_to_research_only:
            raise Exception("You must agree to research-only use.")

        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")

        image = cv2.imread(str(face_image))
        if image is None:
            raise ValueError("Failed to read input image.")
        faces = self.app.get(image)
        if not faces:
            raise ValueError("No face detected.")

        faceid_embeds = torch.tensor(
            faces[0].normed_embedding.tolist(),
            dtype=torch.float32,
            device=self.device,
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
        )

        output_paths: List[Path] = []
        for i, img in enumerate(images):
            out = f"/tmp/out-{i}.png"
            img.save(out)
            output_paths.append(Path(out))

        return output_paths
