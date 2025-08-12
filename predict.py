import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from cog import BasePredictor, Input, Path, BaseModel

class Output(BaseModel):
    portrait: Path

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        self.model_cache = "model_cache"
        os.makedirs(self.model_cache, exist_ok=True)

        # Load base model
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            cache_dir=self.model_cache,
            use_safetensors=True
        ).to("cuda")
        
        # Optimizations
        self.pipe.enable_model_cpu_offload()
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
            description="Random seed", 
            default=None
        ),
    ) -> Output:
        """Generate ancient portrait"""
        try:
            # Load and resize face image
            face_img = Image.open(face_image).convert("RGB").resize((1024, 1024))
            
            # Generate prompt
            prompt = self._generate_prompt(vip_level, profession)
            negative_prompt = "low quality, blurry, bad anatomy, deformed"
            
            # Generate image
            generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None
            output_image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=face_img,
                height=1024,
                width=576,  # 9:16 aspect ratio
                num_inference_steps=30,
                guidance_scale=7,
                generator=generator,
            ).images[0]
            
            output_path = "/tmp/output.png"
            output_image.save(output_path)
            
            return Output(portrait=Path(output_path))
            
        except Exception as e:
            raise RuntimeError(f"Error generating image: {str(e)}")
    
    def _generate_prompt(self, vip_level, profession):
        """Generate detailed prompt"""
        quality_map = {
            "Bronze": "simple",
            "Silver": "elegant",
            "Gold": "luxurious",
            "Platinum": "exquisite",
            "Diamond": "legendary"
        }
        
        class_map = {
            "Swordsman": "martial artist with sword",
            "Blademaster": "warrior with broadsword",
            "Mage": "mage with magical staff",
            "Healer": "healer with herbs",
            "Archer": "archer with longbow",
            "Monk": "monk with prayer beads"
        }
        
        return (
            f"{quality_map[vip_level]} {class_map[profession]}, "
            "ancient Chinese xianxia style, intricate details, "
            "highly detailed, digital painting, 8k resolution"
        )