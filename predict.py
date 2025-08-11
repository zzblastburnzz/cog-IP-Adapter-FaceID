# Prediction interface for Cog ⚙️ (NumPy 2.x–safe)
# - Avoids diffusers.DDIMScheduler (which triggers NumPy at init)
# - Avoids torch.from_numpy (which uses NumPy C-API) by using .tolist() -> torch.tensor
# - Falls back to CPU cleanly (float32) if CUDA libs are missing
# - Tames onnxruntime CPU thread affinity noise by setting thread env vars
#
# Usage: This file is a drop-in replacement for your previous predict.py

from cog import BasePredictor, Input, Path
import os
import cv2
import torch
from PIL import Image
from typing import List
from insightface.app import FaceAnalysis
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,  # uses torch math paths
)

base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
base_cache = "model-cache"
vae_model_path = "stabilityai/sd-vae-ft-mse"
ip_cache = "./ip-cache"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load all models with settings compatible with NumPy 2.x."""
        # Minimize onnxruntime thread-affinity noise
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Pick device & dtypes safely
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Download IP-Adapter FaceID weights if missing
        os.makedirs(ip_cache, exist_ok=True)
        faceid_ckpt = os.path.join(ip_cache, "ip-adapter-faceid_sd15.bin")
        if not os.path.exists(faceid_ckpt):
            os.system(
                f"wget -O {faceid_ckpt} https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin"
            )

        # Face embedding (force CPU provider to avoid CUDA lib issues)
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])  # OR set to ["CUDAExecutionProvider", "CPUExecutionProvider"] if your CUDA libs are present
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # VAE + SD pipeline (no DDIMScheduler to avoid NumPy at init)
        vae = AutoencoderKL.from_pretrained(vae_model_path, torch_dtype=self.torch_dtype)
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=self.torch_dtype,
            vae=vae,
            safety_checker=None,
            feature_extractor=None,
            cache_dir=base_cache,
        )
        # Replace default scheduler with Euler Ancestral (torch math)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

        self.pipe = pipe.to(self.device)

        # IP-Adapter FaceID
        self.ip_model = IPAdapterFaceID(self.pipe, faceid_ckpt, self.device)

    @torch.inference_mode()
    def predict(
        self,
        face_image: Path = Input(description="Input face image"),
        prompt: str = Input(
            description="Input prompt",
            default="photo of a woman in red dress in a garden",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="monochrome, lowres, bad anatomy, worst quality, low quality, blurry, multiple people",
        ),
        width: int = Input(description="Width of output image", default=1024),
        height: int = Input(description="Height of output image", default=1024),
        num_outputs: int = Input(description="Number of images to output", ge=1, le=4, default=1),
        num_inference_steps: int = Input(description="Number of denoising steps", ge=1, le=200, default=30),
        seed: int = Input(description="Random seed. Leave blank to randomize the seed", default=None),
        agree_to_research_only: bool = Input(
            description="You must agree to use this model only for research. It is not for commercial use.",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model (NumPy 2.x–compatible)."""
        if not agree_to_research_only:
            raise Exception(
                "You must agree to use this model for research-only, you cannot use this model comercially."
            )

        # Seed
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")

        # Read & detect face (cv2 returns ndarray but we keep PyTorch ops separate)
        image = cv2.imread(str(face_image))
        if image is None:
            raise ValueError("Failed to read input image.")
        faces = self.app.get(image)
        if not faces:
            raise ValueError("No face detected in the input image.")

        # Convert embedding without torch.from_numpy (safer under NumPy 2.x)
        # Using .tolist() avoids PyTorch<->NumPy C-API path that may be incompatible in some envs
        faceid_embeds = torch.tensor(
            faces[0].normed_embedding.tolist(),  # type: ignore[attr-defined]
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

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

        # Save
        output_paths: List[Path] = []
        for i, img in enumerate(images):
            out = f"/tmp/out-{i}.png"
            img.save(out)
            output_paths.append(Path(out))

        return output_paths
