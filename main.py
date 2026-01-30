import torch
import json
import os
import logging
import random
import warnings
import argparse  # <--- [NEW] Библиотека для чтения команд из консоли
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

# Third-party Libs
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif

# --- 1. SYSTEM MONITORING & LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [ADAPTIVE ENGINE] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

# --- 2. CONFIGURATION OBJECTS ---
@dataclass(frozen=True)
class EngineConfig:
    """Immutable Configuration for the Inference Pipeline."""
    base_model_id: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    motion_adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5-2"
    # Дефолтный путь теперь просто страховка
    default_prompts_path: str = "prompts.json" 
    output_dir: str = "renders"
    default_negative: str = "bad quality, worse quality, low resolution, watermark, text, error, jpeg artifacts"

# --- 3. HARDWARE ABSTRACTION LAYER (HAL) ---
class HardwareProfile:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vram_gb = 0.0
        self.name = "CPU"
        self.compute_capability = 0.0
        
        if self.device == "cuda":
            try:
                props = torch.cuda.get_device_properties(0)
                self.vram_gb = props.total_memory / 1e9
                self.name = torch.cuda.get_device_name(0)
                self.compute_capability = props.major + (props.minor / 10)
            except Exception as e:
                logger.warning(f"Failed to probe CUDA hardware: {e}")

    def is_high_performance_node(self) -> bool:
        return self.vram_gb > 20.0 and self.compute_capability >= 8.0

# --- 4. STRATEGY PATTERN ---
class InferenceStrategy(ABC):
    @abstractmethod
    def configure_pipeline(self, pipe: AnimateDiffPipeline, profile: HardwareProfile):
        pass

    @abstractmethod
    def get_resolution_limit(self) -> Tuple[int, int]:
        pass

class HighPerformanceStrategy(InferenceStrategy):
    def configure_pipeline(self, pipe: AnimateDiffPipeline, profile: HardwareProfile):
        logger.info(f"🚀 Strategy: DATACENTER MODE ({profile.name}).")
        logger.info("⚡ Optimization: Native PyTorch 2.0 SDPA Active.")

    def get_resolution_limit(self) -> Tuple[int, int]:
        return (1024, 1024)

class ConsumerStrategy(InferenceStrategy):
    def configure_pipeline(self, pipe: AnimateDiffPipeline, profile: HardwareProfile):
        logger.info(f"🛡️ Strategy: EFFICIENT MODE ({profile.name}).")
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("⚡ Memory: xFormers Attention Enabled.")
        except Exception:
            logger.warning("⚠️ xFormers not available. Fallback to standard attention.")

        if profile.vram_gb < 15.0:
            logger.info("📉 Memory: Enabling Model CPU Offload.")
            pipe.enable_model_cpu_offload()
        
        pipe.enable_vae_slicing()

    def get_resolution_limit(self) -> Tuple[int, int]:
        return (512, 512)

def strategy_factory(profile: HardwareProfile) -> InferenceStrategy:
    if profile.is_high_performance_node():
        return HighPerformanceStrategy()
    return ConsumerStrategy()

# --- 5. MAIN ENGINE ---
class AdaptiveInferenceEngine:
    def __init__(self, prompts_file: Optional[str] = None):
        """
        Args:
            prompts_file: Path to specific JSON file. If None, uses default from config.
        """
        logger.info("⚙️ Initializing Adaptive Motion Engine v1.0...")
        
        self.config = EngineConfig()
        
        # [NEW] Логика выбора файла
        if prompts_file:
            self.active_prompts_path = prompts_file
            logger.info(f"📂 Custom Prompts File Loaded: {self.active_prompts_path}")
        else:
            self.active_prompts_path = self.config.default_prompts_path
            logger.info(f"📂 Using Default Prompts File: {self.active_prompts_path}")

        self.profile = HardwareProfile()
        self.token = os.getenv("HF_TOKEN")
        
        self.prompts_db = self._load_prompts()
        self.strategy = strategy_factory(self.profile)
        self.pipe = self._build_pipeline()

    def _load_prompts(self) -> Dict[str, Any]:
        if not os.path.exists(self.active_prompts_path):
            logger.warning(f"❌ {self.active_prompts_path} missing. Creating default template.")
            default_data = {"scenes": {}}
            with open(self.active_prompts_path, 'w') as f:
                json.dump(default_data, f, indent=4)
            return default_data
            
        try:
            with open(self.active_prompts_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON format in {self.active_prompts_path}: {e}")
            return {"scenes": {}}

    def _build_pipeline(self) -> AnimateDiffPipeline:
        dtype = torch.float16 if self.profile.device == "cuda" else torch.float32
        
        adapter = MotionAdapter.from_pretrained(
            self.config.motion_adapter_id,
            torch_dtype=dtype,
            token=self.token
        )

        pipe = AnimateDiffPipeline.from_pretrained(
            self.config.base_model_id,
            motion_adapter=adapter,
            torch_dtype=dtype,
            token=self.token
        )

        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, 
            timestep_spacing="trailing", 
            beta_schedule="linear"
        )

        self.strategy.configure_pipeline(pipe, self.profile)
        return pipe

    def generate(self, scene_name: str):
        if scene_name not in self.prompts_db.get('scenes', {}):
            # Если сцены нет, не крашимся, а просто пишем в лог
            return

        scene = self.prompts_db['scenes'][scene_name]
        sys_settings = self.prompts_db.get('system_settings', {})

        max_w, max_h = self.strategy.get_resolution_limit()
        width = min(sys_settings.get('width', 512), max_w)
        height = min(sys_settings.get('height', 512), max_h)

        seed = scene.get('seed', random.randint(0, 2**32-1))
        generator = torch.Generator("cpu").manual_seed(seed)

        logger.info(f"🎬 Rendering: {scene_name} | Res: {width}x{height} | Seed: {seed}")

        output = self.pipe(
            prompt=scene['positive'],
            negative_prompt=scene.get('negative', self.config.default_negative),
            num_frames=16,
            guidance_scale=7.5,
            num_inference_steps=sys_settings.get('base_steps', 25),
            generator=generator,
            width=width,
            height=height,
        )

        os.makedirs(self.config.output_dir, exist_ok=True)
        filename = f"{self.config.output_dir}/{scene_name}_{seed}.gif"
        export_to_gif(output.frames[0], filename)
        logger.info(f"✅ Saved Artifact: {filename}")

# --- 6. ENTRY POINT (CLI INTERFACE) ---
if __name__ == "__main__":
    # 1. Создаем парсер (приемник команд)
    parser = argparse.ArgumentParser(description="Adaptive Motion Lab - CLI Tool")
    
    # 2. Добавляем "гнездо" для аргумента --prompts
    parser.add_argument(
        "--prompts", 
        type=str, 
        default=None, 
        help="Path to the JSON file containing scene definitions (e.g., my_scenes.json)"
    )
    
    # 3. Читаем аргументы
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("❌ No CUDA Device detected. Inference is not possible on this machine.")
    else:
        try:
            # 4. Передаем путь к файлу в движок
            engine = AdaptiveInferenceEngine(prompts_file=args.prompts)
            
            scenes = engine.prompts_db.get('scenes', {})
            
            if not scenes:
                logger.warning("⚠️ No scenes found to render in the provided JSON.")
            
            for name in scenes.keys():
                engine.generate(name)
                
        except KeyboardInterrupt:
            logger.info("🛑 Process Interrupted by User.")
        except Exception as e:
            logger.critical(f"❌ Fatal Error: {e}", exc_info=True)