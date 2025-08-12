import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
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
        """Load the model into memory to make running multiple predictions efficient"""
        
        # Load scheduler
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            VAE_NAME,
            cache_dir=MODEL_CACHE,
            torch_dtype=torch.float16,
        )
        
        # Load base model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            vae=self.vae,
            scheduler=self.scheduler,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
        ).to("cuda")
        
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
        self.faceid_model = load_file(
            FACEID_NAME,
            device="cuda"
        )
        
        # Load LoRA weights for ancient style
        self.pipe.load_lora_weights("path/to/ancient_style_lora", weight_name="ancient_style.safetensors")
        
    def predict(
        self,
        face_image: Path = Input(description="Ảnh khuôn mặt đầu vào"),
        vip_level: str = Input(
            description="Cấp VIP", 
            choices=["Đồng", "Bạc", "Vàng", "Bạch Kim", "Kim Cương"],
            default="Bạc"
        ),
        profession: str = Input(
            description="Nghề nghiệp", 
            choices=["Kiếm Khách", "Đao Khách", "Pháp Sư", "Y Sư", "Cung Thủ", "Tu Sĩ"],
            default="Kiếm Khách"
        ),
        gender: str = Input(
            description="Giới tính",
            choices=["Nam", "Nữ"],
            default="Nam"
        ),
        seed: int = Input(
            description="Random seed", 
            default=None
        ),
    ) -> Output:
        """Run a single prediction on the model"""
        
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        torch.manual_seed(seed)
        
        # Load and process face image
        face_img = load_image(str(face_image))
        face_img = face_img.resize((512, 512))
        
        # Generate prompt based on inputs
        prompt = self._generate_prompt(vip_level, profession, gender)
        negative_prompt = "low quality, blurry, distorted, deformed, extra limbs, bad anatomy"
        
        # Generate image
        output_image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=face_img,
            height=1024,
            width=576,  # 9:16 aspect ratio
            num_inference_steps=30,
            guidance_scale=7,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]
        
        output_path = "/tmp/output.png"
        output_image.save(output_path)
        
        return Output(portrait=Path(output_path))
    
    def _generate_prompt(self, vip_level, profession, gender):
        """Generate detailed prompt based on user inputs"""
        
        # Map VIP level to quality descriptors
        vip_map = {
            "Đồng": "simple, modest",
            "Bạc": "elegant, refined",
            "Vàng": "luxurious, ornate",
            "Bạch Kim": "exquisite, mastercraft",
            "Kim Cương": "legendary, divine"
        }
        
        # Map profession to costume and accessories
        profession_map = {
            "Kiếm Khách": f"{gender} martial artist holding a sword, wearing traditional martial arts robe",
            "Đao Khách": f"{gender} warrior wielding a broadsword, wearing armored robes",
            "Pháp Sư": f"{gender} mage with glowing magical energy, wearing flowing mystical robes",
            "Y Sư": f"{gender} healer with herbal medicine bag, wearing light medicinal robes",
            "Cung Thủ": f"{gender} archer with a longbow, wearing leather and silk hunting outfit",
            "Tu Sĩ": f"{gender} monk with prayer beads, wearing simple but elegant monastic robes"
        }
        
        base_prompt = (
            f"Portrait of a {vip_map[vip_level]} {profession_map[profession]}, "
            "ancient Chinese xianxia style, intricate details, highly detailed, "
            "digital painting, artstation, concept art, smooth, sharp focus, "
            "illustration, unreal engine 5, 8k, cinematic lighting"
        )
        
        return base_prompt