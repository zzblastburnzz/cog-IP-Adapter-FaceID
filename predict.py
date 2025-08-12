import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")
        
        # Enable optimizations
        self.pipe.enable_model_cpu_offload()
        if torch.cuda.is_available():
            self.pipe.enable_xformers_memory_efficient_attention()

    def predict(
        self,
        face_image: Path = Input(description="Input face image"),
        vip_level: str = Input(
            description="VIP level", 
            choices=["Bronze", "Silver", "Gold", "Platinum", "Diamond"],
            default="Silver"
        ),
        profession: str = Input(
            description="Character class", 
            choices=["Swordsman", "Blademaster", "Mage", "Healer", "Archer", "Monk"],
            default="Swordsman"
        ),
        seed: int = Input(
            description="Random seed (leave blank for random)", 
            default=None
        ),
    ) -> Path:
        """Generate ancient Chinese portrait"""
        try:
            # Process input image
            image = Image.open(str(face_image)).convert("RGB").resize((1024, 1024))
            
            # Generate prompt
            prompt = f"{vip_level} {profession}, ancient Chinese style, intricate details, 8k resolution"
            negative_prompt = "low quality, blurry, bad anatomy"
            
            # Generate image
            generator = None
            if seed is not None:
                generator = torch.Generator(device="cuda").manual_seed(seed)
                
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                height=1024,
                width=576,  # 9:16 aspect
                num_inference_steps=30,
                guidance_scale=7,
                generator=generator,
            )
            
            output_path = "/tmp/output.png"
            output.images[0].save(output_path)
            return Path(output_path)
            
        except Exception as e:
            raise RuntimeError(f"Error generating image: {str(e)}")