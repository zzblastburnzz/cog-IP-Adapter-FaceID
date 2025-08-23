import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline
from cog import BasePredictor, Input, Path

# Set environment variables to disable GUI and prevent WPE errors
os.environ['DISPLAY'] = ':0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['GDK_BACKEND'] = 'x11'
os.environ['WPE_BACKEND'] = 'none'
os.environ['WAYLAND_DISPLAY'] = ''

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        # Ensure headless mode
        os.environ['WPE_BACKEND'] = 'none'
        
        print("Loading Stable Diffusion XL model...")
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")
        
        # Enable optimizations
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()
        
        print("Model loaded successfully!")

    def predict(
        self,
        face_image: Path = Input(
            description="Upload your face photo (clear front-facing photo works best)",
            default=None
        ),
        profession: str = Input(
            description="Character class/Profession",
            choices=["Kiếm Khách", "Đao Khách", "Pháp Sư", "Y Sư", "Cung Thủ", "Tu Sĩ"],
            default="Kiếm Khách"
        ),
        vip_level: str = Input(
            description="VIP Level - affects costume quality and lighting effects",
            choices=["Đồng", "Bạc", "Vàng", "Bạch Kim", "Kim Cương"],
            default="Bạc"
        ),
        seed: int = Input(
            description="Random seed (leave empty for random results)",
            default=None
        ),
        guidance_scale: float = Input(
            description="How closely to follow the prompt (7-12 recommended)",
            default=8.0,
            ge=1.0,
            le=20.0
        )
    ) -> Path:
        """Generate ancient Chinese fantasy portrait from face input"""
        try:
            # Load and process input image
            print("Processing input image...")
            init_image = Image.open(str(face_image)).convert("RGB")
            init_image = init_image.resize((576, 1024), Image.LANCZOS)  # 9:16 aspect ratio
            
            # Generate detailed prompt based on inputs
            prompt = self._generate_prompt(profession, vip_level)
            negative_prompt = self._generate_negative_prompt()
            
            print(f"Generating {vip_level} {profession} portrait...")
            print(f"Prompt: {prompt}")
            
            # Set up generator for reproducible results
            generator = None
            if seed is not None:
                generator = torch.Generator(device="cuda").manual_seed(seed)
            
            # Generate the image
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=0.7,  # How much to change the input image
                guidance_scale=guidance_scale,
                num_inference_steps=40,
                generator=generator,
            )
            
            # Save and return the result
            output_path = "/tmp/output.png"
            output.images[0].save(output_path, format="PNG")
            print("Image generation completed successfully!")
            
            return Path(output_path)
            
        except Exception as e:
            error_msg = f"Error generating image: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg)
    
    def _generate_prompt(self, profession, vip_level):
        """Generate detailed prompt based on profession and VIP level"""
        quality_map = {
            "Đồng": "simple cotton robe, basic bronze accessories, soft ambient light",
            "Bạc": "elegant silk robe, silver embroidery, silver jewelry, soft glowing light",
            "Vàng": "luxurious brocade robe, gold embroidery, gold accessories, golden hour lighting",
            "Bạch Kim": "exquisite platinum-trimmed robe, platinum jewelry, divine glowing aura",
            "Kim Cương": "legendary diamond-encrusted armor, radiant diamond accessories, heavenly radiant light beams"
        }
        
        class_map = {
            "Kiếm Khách": "swordmaster holding a gleaming longsword, martial arts stance",
            "Đao Khách": "warrior wielding a massive broadsword, powerful battle pose",
            "Pháp Sư": "mage casting elemental magic, glowing hands, mystical energy aura",
            "Y Sư": "healer holding medicinal herbs, gentle healing glow, compassionate expression",
            "Cung Thủ": "archer drawing an elegant bow, focused aiming pose, forest background",
            "Tu Sĩ": "monk in meditation pose, spiritual aura, peaceful temple background"
        }
        
        return (
            f"Portrait of {class_map[profession]}, {quality_map[vip_level]}, "
            "ancient Chinese xianxia style, fantasy art, intricate details, "
            "highly detailed digital painting, 8k resolution, vertical composition 9:16, "
            "cinematic lighting, beautiful character design, epic fantasy artwork, "
            "artstation trending, unreal engine 5 render"
        )
    
    def _generate_negative_prompt(self):
        """Generate negative prompt to avoid common artifacts"""
        return (
            "low quality, blurry, pixelated, distorted, deformed, bad anatomy, "
            "disfigured, poorly drawn face, mutation, mutated, extra limb, "
            "ugly, poorly drawn hands, poorly drawn feet, poorly drawn face, "
            "out of frame, extra limbs, disfigured, body out of frame, bad hands, "
            "text, error, missing fingers, extra digit, fewer digits, cropped, "
            "worst quality, low quality, normal quality, jpeg artifacts, signature, "
            "watermark, username, blurry, bad proportions, cloned face"
        )

# For local testing
if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    
    # Test with sample image
    test_image_path = "test_face.jpg"  # Replace with actual test image
    if os.path.exists(test_image_path):
        result = predictor.predict(
            face_image=Path(test_image_path),
            profession="Kiếm Khách",
            vip_level="Vàng",
            seed=42
        )
        print(f"Generated image saved to: {result}")
    else:
        print("Test image not found, ready for API deployment")