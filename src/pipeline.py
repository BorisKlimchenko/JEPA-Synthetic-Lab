import torch
import json
import os
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict

# Сторонние библиотеки
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from google.colab import userdata

# --- 1. ЛОГИРОВАНИЕ ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [JEPA-CORE] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- 2. КОНФИГУРАЦИЯ ---
@dataclass
class EngineConfig:
    """Неизменяемая конфигурация путей и моделей."""
    base_model_id: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    motion_adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5-2"
    prompts_path: str = "configs/prompts.json"
    output_dir: str = "renders"

# --- 3. СТРАТЕГИИ ОПТИМИЗАЦИИ ---

class OptimizationStrategy(ABC):
    @abstractmethod
    def apply(self, pipe: AnimateDiffPipeline):
        pass

class HighPerformanceStrategy(OptimizationStrategy):
    """Для мощных GPU (>20GB VRAM)."""
    def apply(self, pipe: AnimateDiffPipeline):
        logger.info("🚀 Strategy: HIGH PERFORMANCE. All systems in VRAM.")
        pipe.enable_vae_slicing()

class SurvivalStrategy(OptimizationStrategy):
    """Для слабых GPU (<16GB VRAM)."""
    def apply(self, pipe: AnimateDiffPipeline):
        logger.info("🛡️ Strategy: SURVIVAL MODE. Aggressive offloading enabled.")
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()

def detect_strategy() -> OptimizationStrategy:
    """Определяет стратегию на основе доступного железа."""
    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not found! CPU mode is not supported efficiently.")
        return SurvivalStrategy()

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    device_name = torch.cuda.get_device_name(0)
    
    logger.info(f"🖥️ Hardware Detected: {device_name} ({vram_gb:.1f} GB)")

    if vram_gb > 20.0:
        return HighPerformanceStrategy()
    else:
        return SurvivalStrategy()

# --- 4. ОСНОВНОЙ ДВИЖОК ---

class LatentMotionEngine:
    """
    SMA-01 Core Engine.
    Управляет загрузкой, оптимизацией и рендерингом.
    """
    
    def __init__(self, hf_token: Optional[str] = None):
        logger.info("⚙️ Initializing Latent Motion Engine...")
        
        self.config = EngineConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Авторизация
        self.token = hf_token or self._fetch_token()
        
        # 1. Загрузка промптов
        self.prompts_db = self._load_prompts()
        
        # 2. Выбор стратегии
        self.strategy = detect_strategy()
        
        # 3. Сборка пайплайна
        self.pipe = self._build_pipeline()

    def _fetch_token(self) -> Optional[str]:
        try:
            return userdata.get('HF_TOKEN')
        except Exception:
            logger.warning("⚠️ HF_TOKEN not found. Using public access.")
            return None

    def _load_prompts(self) -> Dict:
        # При запуске из корня проекта путь должен быть корректным
        if not os.path.exists(self.config.prompts_path):
            logger.warning(f"❌ Config {self.config.prompts_path} not found.")
            return {"scenes": {}}
            
        with open(self.config.prompts_path, 'r') as f:
            data = json.load(f)
        return data

    def _build_pipeline(self) -> AnimateDiffPipeline:
        logger.info("🔌 Loading Neural Network weights...")
        
        adapter = MotionAdapter.from_pretrained(
            self.config.motion_adapter_id,
            torch_dtype=self.dtype,
            token=self.token
        )

        pipe = AnimateDiffPipeline.from_pretrained(
            self.config.base_model_id,
            motion_adapter=adapter,
            torch_dtype=self.dtype,
            token=self.token
        )

        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, 
            timestep_spacing="trailing", 
            beta_schedule="linear"
        )

        self.strategy.apply(pipe)
        return pipe

    def render(self, 
               scene_name: str, 
               num_frames: int = 16, 
               seed: int = -1) -> str:
        """
        Основной метод генерации.
        """
        # 1. Валидация
        if scene_name not in self.prompts_db.get('scenes', {}):
            raise ValueError(f"❌ Scene '{scene_name}' not found in DB.")
            
        scene_data = self.prompts_db['scenes'][scene_name]
        
        # 2. Чтение настроек разрешения из JSON
        sys_config = self.prompts_db.get('system_settings', {})
        width = sys_config.get('width', 512)
        height = sys_config.get('height', 512)
        
        logger.info(f"📐 Resolution set to: {width}x{height}")

        # 3. Сид (Seed)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
            logger.info(f"🎲 Seed Auto-Generated: {seed}")
        else:
            logger.info(f"🔒 Using Fixed Seed: {seed}")
            
        generator = torch.Generator(self.device).manual_seed(seed)
        
        # 4. Инференс
        logger.info(f"🎬 Rendering: {scene_data.get('description', 'Unknown')}")
        
        output = self.pipe(
            prompt=scene_data['positive'],
            negative_prompt=scene_data.get('negative', ""), 
            num_frames=num_frames,
            guidance_scale=7.5,
            num_inference_steps=35,
            generator=generator,
            width=width,
            height=height
        )
        
        # 5. Экспорт
        os.makedirs(self.config.output_dir, exist_ok=True)
        filename = f"{self.config.output_dir}/{scene_name}_{seed}.gif"
        export_to_gif(output.frames[0], filename)
        
        logger.info(f"💾 Artifact saved: {filename}")
        return filename

# Точка входа для проверки
if __name__ == "__main__":
    print("--- JEPA CORE CHECK ---")
    try:
        engine = LatentMotionEngine()
        print("✅ Status: ONLINE.")
    except Exception as e:
        print(f"❌ Status: ERROR: {e}")