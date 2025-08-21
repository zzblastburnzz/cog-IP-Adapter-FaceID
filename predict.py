import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")
        
        # Load additional LoRA for ancient style
        self.pipe.load_lora_weights("ostris/ai-portrait-lora", weight_name="portrait.safetensors")

    def predict(
        self,
        face_image: Path = Input(description="Upload face photo"),
        profession: str = Input(
            description="Character class",
            choices=["Kiếm Khách", "Đao Khách", "Pháp Sư", "Y Sư", "Cung Thủ", "Tu Sĩ"],
            default="Kiếm Khách"
        ),
        vip_level: str = Input(
            description="VIP Level",
            choices=["Đồng", "Bạc", "Vàng", "Bạch Kim", "Kim Cương"],
            default="Bạc"
        ),
        seed: int = Input(description="Random seed", default=None)
    ) -> Path:
        """Generate ancient portrait"""
        try:
            # Process input image
            init_image = Image.open(face_image).convert("RGB").resize((576, 1024))
            
            # Generate prompt
            prompt = self._generate_prompt(profession, vip_level)
            negative_prompt = "low quality, blurry, bad anatomy, deformed"
            
            # Generate image
            generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=0.7,
                guidance_scale=8,
                num_inference_steps=40,
                generator=generator
            )
            
            output_path = "/tmp/output.png"
            output.images[0].save(output_path)
            return Path(output_path)
            
        except Exception as e:
            raise RuntimeError(f"Error: {str(e)}")
    
    def _generate_prompt(self, profession, vip_level):
        """Generate detailed prompt based on inputs"""
        quality_map = {
            "Đồng": "simple costume, basic lighting",
            "Bạc": "elegant costume, soft glow",
            "Vàng": "luxurious outfit, golden light",
            "Bạch Kim": "exquisite robes, divine glow",
            "Kim Cương": "legendary armor, radiant aura"
        }
        
        class_map = {
            "Kiếm Khách": "swordmaster holding a glowing sword",
            "Đao Khách": "warrior with massive blade",
            "Pháp Sư": "mage casting elemental spells",
            "Y Sư": "healer with herbal medicine",
            "Cung Thủ": "archer drawing bow",
            "Tu Sĩ": "monk meditating"
        }
        
        return (
            f"Portrait of {class_map[profession]}, {quality_map[vip_level]}, "
            "ancient Chinese xianxia style, intricate details, fantasy art, "
            "highly detailed, digital painting, 8k resolution, vertical composition"
        )