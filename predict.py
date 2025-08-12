import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, AutoencoderKL
from diffusers.utils import load_image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from safetensors.torch import load_file
from cog import BasePredictor, Input, Path, BaseModel

MODEL_CACHE = "model_cache"
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_NAME = "madebyollin/sdxl-vae-fp16-fix"
IP_ADAPTER_NAME = "h94/IP-Adapter-FaceID"
FACEID_NAME = "h94/IP-Adapter-FaceID/sdxl_faceid_portrait.safetensors"

class Output(BaseModel):
    portrait: Path

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        # Load components with error handling
        try:
            self.scheduler = DDIMScheduler.from_pretrained(
                MODEL_NAME,
                subfolder="scheduler",
                cache_dir=MODEL_CACHE
            )
            
            self.vae = AutoencoderKL.from_pretrained(
                VAE_NAME,
                cache_dir=MODEL_CACHE,
                torch_dtype=torch.float16,
            )
            
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                MODEL_NAME,
                vae=self.vae,
                scheduler=self.scheduler,
                torch_dtype=torch.float16,
                cache_dir=MODEL_CACHE,
                use_safetensors=True,
            ).to("cuda")
            
            # Enable memory efficient attention
            self.pipe.enable_xformers_memory_efficient_attention()
            
            # Load IP Adapter components
            self.image_processor = CLIPImageProcessor.from_pretrained(
                IP_ADAPTER_NAME,
                subfolder="image_processor",
                cache_dir=MODEL_CACHE,
            )
            
            self.image_proj_model = CLIPVisionModelWithProjection.from_pretrained(
                IP_ADAPTER_NAME,
                subfolder="image_projection",
                cache_dir=MODEL_CACHE,
                torch_dtype=torch.float16,
            ).to("cuda")
            
            # Load FaceID model
            faceid_path = os.path.join(MODEL_CACHE, "faceid.safetensors")
            if not os.path.exists(faceid_path):
                faceid_data = load_file(FACEID_NAME)
                torch.save(faceid_data, faceid_path)
            
            # Load LoRA for ancient style (example)
            self.pipe.load_lora_weights(
                "ostris/ai_portrait_lora",
                weight_name="ipadapter_ai_portrait.safetensors",
                adapter_name="ancient_style"
            )
            
        except Exception as e:
            raise RuntimeError(f"Error loading models: {str(e)}")

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
        gender: str = Input(
            description="Gender",
            choices=["Male", "Female"],
            default="Male"
        ),
        seed: int = Input(
            description="Random seed", 
            default=None
        ),
    ) -> Output:
        """Generate ancient portrait"""
        try:
            if seed is None:
                seed = int.from_bytes(os.urandom(2), "big")
            torch.manual_seed(seed)
            
            # Load and process face image
            face_img = load_image(str(face_image))
            face_img = face_img.resize((512, 512))
            
            # Generate prompt
            prompt = self._generate_prompt(vip_level, profession, gender)
            negative_prompt = "low quality, blurry, distorted, deformed, extra limbs, bad anatomy"
            
            # Generate image with error handling
            output_image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                ip_adapter_image=face_img,
                height=1024,
                width=576,
                num_inference_steps=30,
                guidance_scale=7,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).images[0]
            
            output_path = "/tmp/output.png"
            output_image.save(output_path)
            
            return Output(portrait=Path(output_path))
            
        except Exception as e:
            raise RuntimeError(f"Error generating image: {str(e)}")
    
    def _generate_prompt(self, vip_level, profession, gender):
        """Generate detailed prompt"""
        quality_map = {
            "Bronze": "simple, modest quality",
            "Silver": "elegant, refined quality",
            "Gold": "luxurious, ornate quality",
            "Platinum": "exquisite, mastercraft quality",
            "Diamond": "legendary, divine quality"
        }
        
        class_map = {
            "Swordsman": f"{gender} martial artist with sword",
            "Blademaster": f"{gender} warrior with broadsword",
            "Mage": f"{gender} mage with magical staff",
            "Healer": f"{gender} healer with herbal medicine",
            "Archer": f"{gender} archer with longbow",
            "Monk": f"{gender} monk with prayer beads"
        }
        
        return (
            f"{quality_map[vip_level]} {class_map[profession]}, "
            "ancient Chinese xianxia style, intricate details, highly detailed, "
            "digital painting, cinematic lighting, 8k resolution"
        )